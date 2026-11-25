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
"""Reusable shell snippets for NeMo-Gym inline command builders."""

# Reusable bash snippets concatenated into inline command strings built
# by _build_client_cmd() and similar builders.

SHELL_FIND_FREE_PORT = """\
find_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
"""

SHELL_READ_PORT_FILE = """\
read_port_file() {
    local path="$1" name="$2" timeout="${3:-120}"
    local elapsed=0
    while [ ! -f "$path" ]; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            echo "ERROR: $name port file not found after ${timeout}s: $path" >&2
            exit 1
        fi
    done
    cat "$path"
}
"""

SHELL_WAIT_FOR_SERVER = """\
wait_for_server() {
    local url="$1" name="$2" pid="$3" max_attempts="$4" log="$5"
    echo "  Waiting for $name at $url ..."
    for i in $(seq 1 $max_attempts); do
        if curl -s -m 5 "$url" > /dev/null 2>&1; then
            echo "  $name ready after $((i * 5))s"
            return 0
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "ERROR: $name died. Check $log"
            exit 1
        fi
        sleep 5
    done
    echo "ERROR: $name did not start within $((max_attempts * 5))s"
    exit 1
}
"""

SHELL_STOP_BACKGROUND_PROCESS = """\
stop_background_process() {
    local pid="$1" name="${2:-background process}"
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return 0
    fi
    kill -INT "$pid" 2>/dev/null || true
    for _shutdown_attempt in $(seq 1 12); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 5
    done
    echo "WARNING: $name ignored SIGINT; sending SIGTERM" >&2
    kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
        echo "WARNING: $name ignored SIGTERM; sending SIGKILL" >&2
        kill -KILL "$pid" 2>/dev/null || true
    fi
}
"""
