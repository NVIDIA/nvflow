#!/usr/bin/env bash
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: scripts/run_ci_test_lane.sh <minimal|nemo>

  minimal           Install the lightweight CI environment and run tests that
                    must not require NeMo-Skills, NeMo-Run, or Torch. Also
                    validate worker entrypoints on Python 3.13.14.
  nemo              Install the full execution environment and run only tests
                    explicitly marked requires_nemo.
EOF
}

lane=${1:-}
case "$lane" in
  minimal)
    uv sync --locked --only-group ci
    environment_python=$(uv run --no-sync python -c "import sys; print(sys.executable)")
    uv pip install --python "$environment_python" -e . --no-deps --no-build-isolation
    uv run --no-sync python scripts/check_worker_python313.py
    exec uv run --no-sync pytest -v --tb=short -m "not requires_nemo"
    ;;
  nemo)
    uv sync --locked --all-extras --group ci
    uv run --no-sync python -c \
      "import nemo_run; from nemo_skills.inference.model.vllm import process_image_content"
    exec uv run --no-sync pytest -v --tb=short --no-cov -m requires_nemo
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
