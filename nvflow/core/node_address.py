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
"""Backend-neutral shell command for the address a service publishes."""

# Prints the node's first IPv4 address that other machines can reach, or nothing.
# Some GPU nodes list a link-local (169.254.x.x) address first in `hostname -I`,
# so loopback and link-local addresses are skipped and the hostname's own address
# is tried first. Failed lookups print nothing rather than failing, so scripts
# running under `set -euo pipefail` continue to their fallback.
NODE_ADDRESS_COMMAND = (
    "{ getent ahostsv4 \"$(hostname)\" || true; hostname -I 2>/dev/null | tr ' ' '\\n' || true; } "
    "| awk '$1 ~ /^(127|169\\.254)\\./ {next} "
    '$1 ~ /^[0-9.]+$/ && address == "" {address = $1} END {print address}\''
)

__all__ = ["NODE_ADDRESS_COMMAND"]
