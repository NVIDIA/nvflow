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

"""Recipe-independent rubric generation stages and data contracts.

Keep this package initializer free of eager stage imports.  Worker commands
execute modules from the lightweight Python runtime in the NeMo-Skills
container, while stage planning runs in NVFlow's project environment.  Eagerly
importing the planning classes here would unnecessarily pull the latter into
every worker process.
"""

__all__ = [
    "GenerateRubricsStage",
    "NormalizeRubricInputStage",
    "PrepareRubricMethodStage",
    "RubricGenerationStage",
]


def __getattr__(name: str):
    """Load public stage classes only when callers request them."""

    if name == "GenerateRubricsStage":
        from .generate import GenerateRubricsStage

        return GenerateRubricsStage
    if name == "NormalizeRubricInputStage":
        from .normalize import NormalizeRubricInputStage

        return NormalizeRubricInputStage
    if name == "PrepareRubricMethodStage":
        from .prepare_method import PrepareRubricMethodStage

        return PrepareRubricMethodStage
    if name == "RubricGenerationStage":
        from .pipeline import RubricGenerationStage

        return RubricGenerationStage
    raise AttributeError(name)
