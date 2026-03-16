# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Workflow runner for executing stage sequences with dependency management."""

from pathlib import Path

from omegaconf import OmegaConf

from nvflow.core.console import detail, header, info, section, success
from nvflow.core.stage_registry import StageRegistry


class WorkflowRunner:
    """Executes workflow stages with dependency management.

    The WorkflowRunner loads a workflow configuration file and executes
    the specified stages in order, respecting dependencies.

    Supports config inheritance via _base_ key:
        # models/qwen3_14b.yaml
        _base_: ../sft.yaml
        stages:
          sft:
            num_nodes: 32

    Example:
        >>> runner = WorkflowRunner("nvflow/recipes/finance/workflows/sft.yaml")
        >>> runner.run()  # Run all stages
        >>>
        >>> # Or run specific stage (short name from config)
        >>> runner.run(stages=["sft"])
    """

    def __init__(self, config_path: str):
        """Initialize the workflow runner.

        Args:
            config_path: Path to workflow configuration YAML file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config with inheritance support and resolve interpolations
        config = self._load_config_with_inheritance(self.config_path)
        self.config = OmegaConf.to_container(config, resolve=True)

        # Expand dynamic sections (models, checkpoints) into stage configs
        self._expand_dynamic_stages()

        # Extract context for hierarchical registry
        self.recipe = self.config["recipe"]
        self.workflow_name = self.config["workflow"]["name"]
        self.workflow_type = self.config["workflow"].get("type", "unknown")
        self.cluster = self.config["cluster"]

    def _load_config_with_inheritance(self, config_path: Path) -> OmegaConf:
        """Load config with _base_ inheritance support.

        If the config contains a _base_ key, recursively loads and merges
        the base config. Child config values override base config values.

        Args:
            config_path: Path to the config file

        Returns:
            Merged OmegaConf configuration
        """
        config = OmegaConf.load(config_path)

        if "_base_" in config:
            # Resolve base path relative to current config file
            base_path = (config_path.parent / config["_base_"]).resolve()

            if not base_path.exists():
                raise FileNotFoundError(
                    f"Base config not found: {config['_base_']} (resolved to {base_path})"
                )

            # Recursively load base config (supports chained inheritance)
            base_config = self._load_config_with_inheritance(base_path)

            # Remove _base_ key before merging
            config = OmegaConf.to_container(config)
            del config["_base_"]
            config = OmegaConf.create(config)

            # Deep merge: child overrides base
            config = OmegaConf.merge(base_config, config)

        return config

    # ------------------------------------------------------------------
    # Dynamic stage expansion
    # ------------------------------------------------------------------

    # Top-level YAML sections that can generate stages.  Each entry in
    # ``pipeline_stages`` is checked against these sections; if found,
    # the raw entry config is copied into ``stages`` as-is.
    _EXPANDABLE_SECTIONS = ("models", "checkpoints")

    def _expand_dynamic_stages(self) -> None:
        """Expand top-level sections into stage configs.

        For each entry in ``pipeline_stages`` that is not already defined
        in ``stages`` but exists in an expandable section (``models``,
        ``checkpoints``), creates a stage config by copying the entry
        as-is and tagging it with source metadata.  The registered stage
        class is responsible for interpreting the raw config.

        Checkpoint entries with ``eval_steps`` are additionally expanded
        so that one pipeline stage per step is created (e.g.
        ``sft-run`` with ``eval_steps: [1000, 2000]`` becomes
        ``sft-run-1000`` and ``sft-run-2000``).
        """
        for section_name in self._EXPANDABLE_SECTIONS:
            section = self.config.get(section_name, {})
            if not section:
                continue

            if section_name == "checkpoints":
                self._expand_checkpoint_pipeline_stages(section)

            expanded = 0
            for entry_name, entry_config in section.items():
                if not isinstance(entry_config, dict):
                    continue

                stage_entries = self._stage_entries_for(section_name, entry_name, entry_config)

                for stage_name, stage_config in stage_entries:
                    if stage_name in self.config.get("stages", {}):
                        continue
                    self.config.setdefault("stages", {})[stage_name] = stage_config
                    expanded += 1

            if expanded:
                info(f"Expanded {expanded} stage(s) from {section_name} section")

    # Keys that are structural to the runner and should not be forwarded
    # as workflow-level defaults into expanded stage configs.
    _STRUCTURAL_KEYS = frozenset(
        {
            "recipe",
            "workflow",
            "cluster",
            "pipeline_stages",
            "stages",
            *_EXPANDABLE_SECTIONS,
        }
    )

    def _workflow_defaults(self) -> dict:
        """Collect workflow-level config to forward as defaults.

        Returns all top-level keys that are not structural (pipeline
        plumbing) so that expanded stages can access workflow-wide
        settings like ``base_output_dir`` or ``benchmarks`` without
        the runner knowing what those keys mean.
        """
        return {k: v for k, v in self.config.items() if k not in self._STRUCTURAL_KEYS}

    def _stage_entries_for(
        self,
        section_name: str,
        entry_name: str,
        entry_config: dict,
    ) -> list[tuple[str, dict]]:
        """Build ``(stage_name, stage_config)`` pairs for one section entry.

        For ``models``: one stage per model, config copied as-is.
        For ``checkpoints``: one stage per ``eval_steps`` entry, with
        ``_step`` metadata attached.

        Workflow-level defaults are merged first so that entry-specific
        values take precedence.
        """
        base = {
            **self._workflow_defaults(),
            **entry_config,
            "_source_section": section_name,
            "_source_name": entry_name,
        }

        if section_name != "checkpoints":
            return [(entry_name, base)]

        eval_steps = entry_config.get("eval_steps", [])
        if isinstance(eval_steps, int):
            eval_steps = [eval_steps]

        return [(f"{entry_name}-{step}", {**base, "_step": step}) for step in eval_steps]

    def _expand_checkpoint_pipeline_stages(self, checkpoints: dict) -> None:
        """Replace checkpoint run names in ``pipeline_stages`` with per-step names."""
        expanded = []
        for stage in self.config.get("pipeline_stages", []):
            if stage in checkpoints:
                run_config = checkpoints[stage]
                eval_steps = run_config.get("eval_steps", [])
                if isinstance(eval_steps, int):
                    eval_steps = [eval_steps]
                for step in eval_steps:
                    expanded.append(f"{stage}-{step}")
            else:
                expanded.append(stage)
        self.config["pipeline_stages"] = expanded

    def run(self, stages: list[str] | None = None) -> None:
        """Run workflow stages.

        Args:
            stages: List of stage names to run. If None, runs all stages
                   defined in config's pipeline_stages.

        Example:
            >>> runner.run()  # Run all stages
            >>> runner.run(stages=["download"])  # Run one stage
            >>> runner.run(stages=["download", "validate"])  # Run multiple
        """
        all_stages = self.config["pipeline_stages"]
        stages_to_run = stages if stages else all_stages

        # Validate that requested stages exist in config
        self._validate_stages(stages_to_run, all_stages)

        header(f"NVFlow - Running Workflow: {self.workflow_name}")
        detail("Workflow Type", self.workflow_type)
        detail("Cluster", self.cluster)
        detail("Stages to run", f"{len(stages_to_run)}/{len(all_stages)}")

        # Execute stages
        completed_stages = []
        for stage_name in stages_to_run:
            self._run_stage(stage_name)
            completed_stages.append(stage_name)

        header("✅ Workflow Complete!")
        success(f"Completed {len(completed_stages)} stage(s): {', '.join(completed_stages)}")

    def _run_stage(self, stage_name: str) -> None:
        """Run a single stage.

        Args:
            stage_name: Short stage name (e.g., "sft", "generate_qa", "download")
                       Stage is resolved using recipe and workflow context
        """
        section(f"Running Stage: {stage_name}")

        # Get stage configuration
        stage_config = self.config["stages"][stage_name]

        # Get stage class from hierarchical registry with explicit context
        if not StageRegistry.has(self.recipe, self.workflow_name, stage_name):
            raise ValueError(
                f"Stage '{stage_name}' not found in registry at "
                f"{self.recipe}.{self.workflow_name}.{stage_name}. "
                f"Make sure the stage is properly registered with "
                f"@StageRegistry.register(recipe='{self.recipe}', "
                f"workflow='{self.workflow_name}', stage='{stage_name}')"
            )

        stage_class = StageRegistry.get(self.recipe, self.workflow_name, stage_name)
        stage = stage_class()

        # Generate experiment name for this stage
        expname = self._get_expname(stage_name, stage_config)

        # Get dependencies (other stages this stage depends on)
        dependencies = stage_config.get("dependencies", [])
        run_after = (
            [self._get_expname(dep, self.config["stages"][dep]) for dep in dependencies]
            if dependencies
            else None
        )

        if dependencies:
            info(f"Dependencies: {', '.join(dependencies)}")

        # Validate stage configuration
        stage.validate_config(stage_config)

        # Execute stage
        stage.execute(
            config=stage_config, cluster=self.cluster, expname=expname, run_after=run_after
        )

        success(f"Stage '{stage_name}' completed")

    def _get_expname(self, stage_name: str, stage_config: dict) -> str:
        """Generate clean experiment name for a stage.

        With hierarchical registry, stage names are short (e.g., "sft"),
        so we construct clean experiment names without redundancy.

        Args:
            stage_name: Short stage name (e.g., "sft", "generate_qa", "download")
            stage_config: Configuration dict for this stage

        Returns:
            Experiment name formats:
            - With run_name: "workflow-stage-run_name" (e.g., "training_sft-sft-lr5e6-bs128")
            - Without run_name: "workflow-stage" (e.g., "training_sft-sft")
        """
        # Base name: workflow + stage (no recipe prefix needed in expname)
        base_name = f"{self.workflow_name}-{stage_name}"

        # Append run_name if specified (for config experiments)
        run_name = stage_config.get("run_name")
        if run_name:
            return f"{base_name}-{run_name}"

        return base_name

    def _validate_stages(self, stages_to_run: list[str], all_stages: list[str]) -> None:
        """Validate that requested stages exist and are registered.

        Args:
            stages_to_run: List of stage names to run
            all_stages: List of all stages defined in config

        Raises:
            ValueError: If any stage is invalid or not registered
        """
        for stage in stages_to_run:
            # Check if stage is in the workflow config
            if stage not in all_stages:
                raise ValueError(
                    f"Stage '{stage}' not found in workflow config. Available stages: {all_stages}"
                )

            # Check if stage is registered in hierarchical registry
            if not StageRegistry.has(self.recipe, self.workflow_name, stage):
                raise ValueError(
                    f"Stage '{stage}' not registered at {self.recipe}.{self.workflow_name}.{stage}. "
                    f"Make sure to import the module containing the stage definition and "
                    f"that it's registered with @StageRegistry.register(recipe='{self.recipe}', "
                    f"workflow='{self.workflow_name}', stage='{stage}')"
                )

            # Check if stage has configuration
            if stage not in self.config["stages"]:
                raise ValueError(
                    f"Stage '{stage}' listed in pipeline_stages but no configuration "
                    f"found in stages section"
                )

    def validate_config(self) -> None:
        """Validate the workflow configuration.

        Checks for:
        - Required fields in config
        - Valid stage definitions
        - Proper dependency chains

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required top-level fields
        required_fields = ["recipe", "workflow", "cluster", "pipeline_stages", "stages"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in config: {field}")

        # Validate all stages
        all_stages = self.config["pipeline_stages"]
        self._validate_stages(all_stages, all_stages)

        success(f"Configuration valid: {self.config_path}")
