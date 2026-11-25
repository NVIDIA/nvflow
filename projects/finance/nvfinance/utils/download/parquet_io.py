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
"""Crash-safe writes for the filing metadata parquet."""

import os

import pandas as pd


def write_parquet_atomic(frame: pd.DataFrame, path: str) -> None:
    """Replace the parquet at *path* only once the new one is complete on disk.

    Writing in place would truncate the file first, so an interrupted write
    leaves no parquet footer and the result cannot be read at all. Nothing can
    rebuild this file: the EDGAR url, cik, filing_date and company_name it holds
    exist nowhere else on disk.
    """
    temporary_path = f"{path}.partial"
    try:
        frame.to_parquet(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
