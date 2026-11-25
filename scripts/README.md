# Repository scripts

Root scripts are limited to platform maintenance, generic operations, tracing,
and Python debugging:

- `run_flow.py` — compatibility shim for `python -m nvflow.cli.run_workflow`
- `materialize_git_snapshot.py` — release-image Git/submodule snapshot adapter
- `check_*.py` — repository policy hooks
- `run_ci_test_lane.sh` — shared CI test-lane driver
- `trivy_report_diff.py` — applies the fixable HIGH/CRITICAL policy to Trivy
  JSON reports, optionally against a baseline report
- `validate_trivy_image_report.py` — checks that a Trivy image report covers
  the expected platform
- `update_trivy_issue.py` — keeps a remediation issue in sync with scheduled
  Trivy findings
- `build_images.sh` — builds, pushes, and verifies the custom container images
- `setup_containers.sh` and `monitor_gpu.sh` — generic cluster operations
- `ray-test/` — optional test-only supplied Ray cluster deployment on Slurm
- `view_traces.py` — generic trace inspection

Finance-specific analysis and conversion tools live under
[`projects/finance/scripts`](../projects/finance/scripts/). Retired fixture
generators are intentionally absent from the repository.

Root `pyproject.toml` is the only dependency-policy source of truth. After
changing its execution, override, or constraint lists, regenerate the bundled
managed-environment files and NeMo-Skills worker-image pin with:

```bash
uv run python scripts/check_project_environment_policy.py --write
```

Custom multi-architecture image builds, including the routed vLLM v0.29.0 image,
are declared in [`../docker-bake.hcl`](../docker-bake.hcl).

The Python workflow entry stays in the caller's interpreter for debugger use:

```bash
uv run python -m pudb.run -m nvflow.cli.run_workflow \
  training --workflow finance:sft/qwen3_4b --cluster my_cluster
```
