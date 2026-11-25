#!/usr/bin/env bash
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
# TEST INFRASTRUCTURE ONLY. This allocation supplies an ordinary Ray Jobs
# endpoint to NVFlow; NVFlow does not own or invoke this script.

#SBATCH --job-name=nvflow-ray-test
#SBATCH --output=nvflow-ray-test-%j.out

set -Eeuo pipefail

: "${ROLE:?set ROLE to the target runtime role}"
: "${IMAGE:?set IMAGE to an enroot/pyxis .sqsh path}"
: "${IMAGE_ID:?set IMAGE_ID to the immutable target identity}"
: "${ENDPOINT_FILE:?set ENDPOINT_FILE on shared storage}"
: "${ARTIFACT_ROOT:?set ARTIFACT_ROOT on shared storage}"
: "${NVFLOW_INSECURE_TEST_ENDPOINT_ACK:?set to I_UNDERSTAND_RAY_JOBS_HAS_NO_AUTH}"
if [ "${NVFLOW_INSECURE_TEST_ENDPOINT_ACK}" != "I_UNDERSTAND_RAY_JOBS_HAS_NO_AUTH" ]; then
    echo "ERROR: invalid NVFLOW_INSECURE_TEST_ENDPOINT_ACK" >&2
    exit 1
fi

RAY_BIN_DIR="${RAY_BIN_DIR:-}"
MOUNTS="${MOUNTS:-${ARTIFACT_ROOT}:${ARTIFACT_ROOT}}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-300}"
# NVFlow submits every ready step and lets the cluster queue them, but Ray fails a
# Job that waits longer than this for resources (15 minutes by default).
RAY_JOB_START_TIMEOUT_SECONDS="${RAY_JOB_START_TIMEOUT_SECONDS:-604800}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [ "${#nodes[@]}" -eq 0 ]; then
    echo "ERROR: Slurm allocation contains no nodes" >&2
    exit 1
fi

# The address other hosts reach a node at comes from its name, as in NeMo-RL's
# ray.sub; `hostname -I` lists a link-local interface first on GPU nodes.
node_ip() {
    { getent ahostsv4 "$1" || true; } | awk '$1 !~ /^(127|169\.254)\./ {print $1; exit}'
}

# Shared nodes may already use any fixed port, and Ray then fails quietly: with its
# agent's port taken, a cluster answers but accepts no job. Other hosts must know the
# GCS and dashboard ports, and Ray refuses an OS-assigned client server port, so pick
# these three free, below the kernel's ephemeral range, where outgoing connections
# cannot take them; the OS assigns every other port.
free_port() {
    local port="$1" limit
    read -r limit _ 2>/dev/null </proc/sys/net/ipv4/ip_local_port_range || limit=32768
    [ "${limit}" -gt "${port}" ] || limit=65536
    while [ "${port}" -lt "${limit}" ]; do
        if [ -z "$(ss -Hltn "sport = :${port}")" ]; then
            echo "${port}"
            return
        fi
        port=$((port + 1))
    done
    return 1
}
os_ports="--dashboard-agent-listen-port=0 --min-worker-port=0 --max-worker-port=0"

head_node="${nodes[0]}"
head_ip="$(node_ip "${head_node}")"
if [ -z "${head_ip}" ]; then
    echo "ERROR: could not resolve a routable IPv4 address for ${head_node}" >&2
    exit 1
fi
# This script runs on the head node. The job ID offset keeps clusters that start
# together on one node from choosing the same ports.
if ! RAY_PORT="$(free_port $((20000 + SLURM_JOB_ID % 3000 * 3)))" ||
    ! DASHBOARD_PORT="$(free_port $((RAY_PORT + 1)))" ||
    ! CLIENT_PORT="$(free_port $((DASHBOARD_PORT + 1)))"; then
    echo "ERROR: no free port for Ray on ${head_node}" >&2
    exit 1
fi

mkdir -p "$(dirname "${ENDPOINT_FILE}")" "${ARTIFACT_ROOT}"
rm -f "${ENDPOINT_FILE}"

container_prefix=""
if [ -n "${RAY_BIN_DIR}" ]; then
    container_prefix="export PATH='${RAY_BIN_DIR}':\"\${PATH}\";"
fi
# Keep the name pattern in sync with CREDENTIAL_NAME in clusters.py.
credential_unset() {
    local names
    if [ "${NVFLOW_TEST_PASS_CREDENTIALS:-}" = "1" ]; then
        echo "WARNING: Ray test cluster inherits credential-like variables;" \
            "anyone who can reach its Jobs API can read them" >&2
        return
    fi
    names="$(
        env | awk -F= '
          $1 ~ /(TOKEN|API_KEY|SECRET|PASSWORD|CREDENTIAL)$/ &&
          $1 != "NVFLOW_INSECURE_TEST_ENDPOINT_ACK" { print $1 }
        ' | tr '\n' ' '
    )"
    if [ -n "${names// /}" ]; then
        echo "Ray test cluster will not inherit credential-like variables: ${names}" >&2
        printf 'unset %s;' "${names}"
    fi
}
secret_unset="$(credential_unset)"

# Every image puts python on PATH, but not every image links Ray's console
# script, so start Ray through its CLI module.
head_command="${container_prefix}${secret_unset}
export NVFLOW_IMAGE_ID='${IMAGE_ID}' PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export RAY_JOB_START_TIMEOUT_SECONDS='${RAY_JOB_START_TIMEOUT_SECONDS}'
exec python -m ray.scripts.scripts start --head --block --disable-usage-stats \
  --node-ip-address='${head_ip}' --port='${RAY_PORT}' \
  --dashboard-host=0.0.0.0 --dashboard-port='${DASHBOARD_PORT}' \
  --ray-client-server-port='${CLIENT_PORT}' ${os_ports}"

srun --overlap --nodes=1 --ntasks=1 --nodelist="${head_node}" \
    --container-image="${IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-name="nvflow-ray-${ROLE}-${SLURM_JOB_ID}-head" \
    --no-container-mount-home \
    bash -lc "${head_command}" &
head_step_pid=$!

deadline=$((SECONDS + START_TIMEOUT_SECONDS))
# The version endpoint answers even when Ray's job agent is down; only an accepted
# job shows that the cluster can run work.
until curl --fail --silent --show-error \
    "http://${head_ip}:${DASHBOARD_PORT}/api/version" >/dev/null 2>&1 &&
    curl --fail --silent --show-error -X POST -H "Content-Type: application/json" \
        -d "{\"entrypoint\": \"true\", \"submission_id\": \"harness-ready-${SECONDS}\"}" \
        "http://${head_ip}:${DASHBOARD_PORT}/api/jobs/" >/dev/null 2>&1; do
    if ! kill -0 "${head_step_pid}" 2>/dev/null; then
        echo "ERROR: Ray head step exited before the Jobs API became ready" >&2
        wait "${head_step_pid}"
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "ERROR: Ray Jobs API did not accept a job within ${START_TIMEOUT_SECONDS}s" >&2
        exit 1
    fi
    sleep 2
done

worker_pids=()
for worker_node in "${nodes[@]:1}"; do
    worker_ip="$(node_ip "${worker_node}")"
    if [ -z "${worker_ip}" ]; then
        echo "ERROR: could not resolve a routable IPv4 address for ${worker_node}" >&2
        exit 1
    fi
    worker_command="${container_prefix}${secret_unset}
export NVFLOW_IMAGE_ID='${IMAGE_ID}' PYTHONUNBUFFERED=1 UV_NO_SYNC=1
exec python -m ray.scripts.scripts start --block --disable-usage-stats \
  --node-ip-address='${worker_ip}' --address='${head_ip}:${RAY_PORT}' ${os_ports}"
    srun --overlap --nodes=1 --ntasks=1 \
        --nodelist="${worker_node}" \
        --container-image="${IMAGE}" \
        --container-mounts="${MOUNTS}" \
        --container-name="nvflow-ray-${ROLE}-${SLURM_JOB_ID}-${worker_node}" \
        --no-container-mount-home \
        bash -lc "${worker_command}" &
    worker_pids+=("$!")
done

temporary="${ENDPOINT_FILE}.tmp.${SLURM_JOB_ID}"
cat >"${temporary}" <<EOF
ROLE=${ROLE}
JOBS_URL=http://${head_ip}:${DASHBOARD_PORT}
CONTROLLER_JOBS_URL=http://${head_ip}:${DASHBOARD_PORT}
IMAGE_ID=${IMAGE_ID}
SLURM_JOB_ID=${SLURM_JOB_ID}
EOF
mv -f "${temporary}" "${ENDPOINT_FILE}"

echo "Ray test cluster ready: role=${ROLE} jobs_url=http://${head_ip}:${DASHBOARD_PORT}"
echo "Cancel with: scancel ${SLURM_JOB_ID}"

wait "${head_step_pid}" "${worker_pids[@]}"
