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
"""NVFlow, the workflow orchestration platform.

NVFlow plans and runs NeMo-based data generation, training, and evaluation
workflows built from project stages and NVFlow's shared stages.
"""

from typing import TYPE_CHECKING, Any

try:
    from nvflow._version import __version__
except ImportError:
    # Fallback for editable installs or when _version.py hasn't been generated
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from nvflow.core.workflow_definition import WorkflowDefinition, load_workflow

__all__ = ["WorkflowDefinition", "__version__", "load_workflow"]


def __getattr__(name: str) -> Any:
    """Load workflow authoring APIs without burdening worker imports."""
    if name in {"WorkflowDefinition", "load_workflow"}:
        from nvflow.core.workflow_definition import WorkflowDefinition, load_workflow

        return {
            "WorkflowDefinition": WorkflowDefinition,
            "load_workflow": load_workflow,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
