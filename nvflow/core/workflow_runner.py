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

import sys
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

    def _before_run(self) -> None:
        """Executor-specific setup hook called at the start of :meth:`run`.

        Default implementation applies the Slurm sbatch-args autopatch so
        cluster-level ``extra_sbatch_args`` reach every Slurm submission.
        Installed lazily here (not at CLI startup) because importing
        nemo_skills.pipeline pulls in torch/transformers (~15s cold cache).

        Subclasses (e.g. :class:`~nvflow.core.ray_workflow_runner.RayWorkflowRunner`)
        override this to skip the Slurm patch or perform Ray-specific setup.
        """
        from nvflow.lib.sbatch import apply_sbatch_args_autopatch

        apply_sbatch_args_autopatch()

    def run(
        self,
        stages: list[str] | None = None,
        environment: list[str] | None = None,
    ) -> None:
        """Run workflow stages.

        Args:
            stages: List of stage names to run. If None, runs all stages
                   defined in config's pipeline_stages.
            environment: Optional list of environment names to run for.
                   If None, runs all environments defined in config.

        Example:
            >>> runner.run()  # Run all stages, all environments
            >>> runner.run(stages=["download"])  # Run one stage
            >>> runner.run(environment=["equivalence_llm_judge"])  # Single env
            >>> runner.run(environment=["mcqa", "equivalence_llm_judge"])
        """
        all_stages = self.config["pipeline_stages"]
        stages_to_run = stages if stages else all_stages

        # Validate that requested stages exist in config
        self._validate_stages(stages_to_run, all_stages)

        # Executor-specific pre-run setup (Slurm sbatch-args patch by default).
        self._before_run()

        # Warn about sibling stages that are declared in pipeline_stages
        # but not currently registered (e.g., their import failed).
        self._preflight_pipeline_health(all_stages, stages_to_run)

        if environment:
            environments = self.config.get("environments", {})
            for env_name in environment:
                if env_name not in environments:
                    available = ", ".join(environments.keys())
                    raise ValueError(f"Unknown environment '{env_name}'. Available: {available}")

        header(f"NVFlow - Running Workflow: {self.workflow_name}")
        detail("Workflow Type", self.workflow_type)
        detail("Cluster", self.cluster)
        detail("Stages to run", f"{len(stages_to_run)}/{len(all_stages)}")
        if environment:
            detail("Environment", ", ".join(environment))

        # Execute stages
        completed_stages = []
        for stage_name in stages_to_run:
            self._run_stage(stage_name, environment=environment, stages_to_run=stages_to_run)
            completed_stages.append(stage_name)

        header("✅ Workflow Complete!")
        success(f"Completed {len(completed_stages)} stage(s): {', '.join(completed_stages)}")

    def _preflight_pipeline_health(
        self,
        all_stages: list[str],
        stages_to_run: list[str],
    ) -> None:
        """Warn when pipeline_stages contains unregistered stages.

        Subset runs only validate the requested stages, so siblings that
        failed to import sit unnoticed until the user runs the full
        pipeline.  This method surfaces them upfront on stderr (warn-only,
        does not block the run).
        """
        deferred = [s for s in all_stages if s not in stages_to_run]
        missing = [s for s in deferred if not StageRegistry.has(self.recipe, self.workflow_name, s)]
        if missing:
            print(
                "[nvflow] WARNING: pipeline declares stages that are not "
                "currently registered (other stages may have failed to "
                "import; check earlier discovery warnings).  The workflow "
                "WILL NOT complete end-to-end without fixing these:",
                file=sys.stderr,
            )
            for s in missing:
                print(f"  - {self.recipe}.{self.workflow_name}.{s}", file=sys.stderr)

    def _run_stage(
        self,
        stage_name: str,
        environment: list[str] | None = None,
        stages_to_run: list[str] | None = None,
    ) -> None:
        """Run a single stage.

        Args:
            stage_name: Short stage name (e.g., "sft", "generate_qa", "download")
                       Stage is resolved using recipe and workflow context
            environment: Optional list of environment names to filter to.
            stages_to_run: Stages being submitted in this session.  Slurm
                deps are only wired for stages in this list; cross-session
                deps are dropped because nemo-run's job directory may not
                contain their records.
        """
        section(f"Running Stage: {stage_name}")

        # Get stage configuration and inject environment filter
        stage_config = {**self.config["stages"][stage_name]}
        if environment is not None:
            # `-e` is a filter, never an expansion: a stage that declares an
            # ``environments`` scope only runs for the requested env(s) within
            # that scope. A scoped stage with no in-scope env is skipped (it
            # never declared that env, so forcing it would read stale/foreign
            # source_data). Mirrors ``_resolve_env_names`` so dependency wiring
            # and execution agree on a scoped stage's env set.
            selected = self._filter_stage_environment(stage_config, environment)
            if not selected:
                info(
                    f"Skipping stage '{stage_name}' for "
                    f"{', '.join(environment)} (outside its environments scope)"
                )
                return
            stage_config["_environment"] = selected

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

        # Only wire Slurm deps for stages submitted in this session.
        dependencies = stage_config.get("dependencies", [])
        if stages_to_run is not None:
            dependencies = [d for d in dependencies if d in stages_to_run]
        run_after = self._get_run_after_names(dependencies, environment)

        if dependencies:
            info(f"Dependencies: {', '.join(dependencies)}")

        # Validate stage configuration
        stage.validate_config(stage_config)

        # Execute stage on its resolved cluster (per-stage CPU/GPU routing
        # for 2-cluster Ray; no-op on Slurm / single-URL Ray).
        stage.execute(
            config=stage_config,
            cluster=self._resolve_stage_cluster(stage_config),
            expname=expname,
            run_after=run_after,
        )

        success(f"Stage '{stage_name}' completed")

    def _resolve_stage_cluster(self, stage_config: dict):
        """Return the workflow cluster unchanged (Slurm path — no per-stage routing).

        Ray-specific 2-cluster routing lives in
        :class:`~nvflow.core.ray_workflow_runner.RayWorkflowRunner`, which
        overrides this method.
        """
        return self.cluster

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

    def _get_run_after_names(
        self,
        dependencies: list[str],
        environment: list[str] | None,
    ) -> list[str] | None:
        """Build ``run_after`` experiment names for Slurm dependency tracking.

        Per-environment stages submit jobs with ``{expname}-{env_name}``
        suffixes.  This method expands dependency names to match those
        suffixed experiment names so that ``nemo-run`` can resolve the
        correct Slurm job handles.

        For stages without ``environments``, the base experiment name is
        used (unchanged from previous behaviour).
        """
        if not dependencies:
            return None
        names: list[str] = []
        for dep in dependencies:
            dep_config = self.config["stages"][dep]
            base = self._get_expname(dep, dep_config)
            if dep_config.get("environments"):
                env_names = self._resolve_env_names(dep_config, environment)
                names.extend(f"{base}-{env}" for env in env_names)
            else:
                names.append(base)
        return names or None

    @staticmethod
    def _resolve_env_names(
        stage_config: dict,
        environment: list[str] | None,
    ) -> list[str]:
        """Return the environment names a stage will iterate over.

        Mirrors the filtering logic of ``resolve_environments()`` in
        ``nvflow.lib.rl.helpers`` but operates on the raw config dict
        so the core module stays independent of recipe-specific code.
        """
        envs = stage_config.get("environments", {})
        if not envs:
            return []
        if environment:
            return [e for e in environment if e in envs]
        return list(envs.keys())

    @staticmethod
    def _filter_stage_environment(
        stage_config: dict,
        environment: list[str],
    ) -> list[str]:
        """Filter the CLI ``-e`` env(s) by a stage's own ``environments`` scope.

        ``-e`` is a filter, not an expansion. A stage that declares an
        ``environments`` block runs only for the requested env(s) that fall
        within that block (the intersection); an empty result means the stage
        is out of scope for every requested env and the caller should skip it.
        A stage with no ``environments`` block is unscoped and runs for the
        requested env(s) unchanged. Mirrors :meth:`_resolve_env_names` so
        dependency wiring and execution agree on a scoped stage's env set.
        """
        stage_envs = stage_config.get("environments")
        if not stage_envs:
            return environment
        return [e for e in environment if e in stage_envs]

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

        # Walk the transitive dependency graph of stages_to_run and verify
        # each dependency is both configured and registered.  Without this,
        # the runner builds Slurm --dependency names for stages that were
        # never submitted (their import failed silently).
        closure: set[str] = set(stages_to_run)
        queue: list[str] = list(stages_to_run)
        while queue:
            s = queue.pop()
            for d in self.config["stages"].get(s, {}).get("dependencies", []):
                if d in closure:
                    continue
                closure.add(d)
                queue.append(d)
                if d not in self.config["stages"]:
                    raise ValueError(
                        f"Stage '{s}' depends on '{d}' which has no config block in 'stages:'."
                    )
                if not StageRegistry.has(self.recipe, self.workflow_name, d):
                    raise ValueError(
                        f"Stage '{s}' depends on '{d}' which is not registered "
                        f"at {self.recipe}.{self.workflow_name}.{d}. Check "
                        "earlier discovery warnings on stderr for the "
                        "underlying import failure."
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
