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

usage() {
  cat <<'EOF'
Usage: REGISTRY=<registry-host>/<namespace> scripts/build_images.sh [--serial] [--platform PLATFORM]
         [--log-dir DIR] [TARGET ...]

Build and push NVFlow's custom images for linux/amd64 and linux/arm64, one Bake
target at a time. Logs each build, verifies each pushed manifest, and stops at
the first failure.

  TARGET      Bake targets, in order (default: nemo-skills nemo-rl nemo-gym vllm-0-29-0
              sglang-0-5-17 nvflow-client)
  --serial    Also build one platform at a time, then join both into the target's
              tag. Use it on hosts with limited CPU or memory.
  --platform  Build only linux/amd64 or linux/arm64, for quick testing. Pushes
              <tag>-<arch> only, so it never replaces a dual-architecture tag.
  --log-dir   Log directory outside the clone (default: ~/nvflow-build-logs/<timestamp>)

Environment:
  REGISTRY    Registry and namespace to push to (required)
  BUILDER     Buildx builder name (default: nvflow-multiarch; created if missing)
  NVFLOW_TAG  nvflow-client tag (default: v1.2.0)
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

builder=${BUILDER:-nvflow-multiarch}
log_dir="$HOME/nvflow-build-logs/$(date +%Y%m%d-%H%M%S)"
platforms=(linux/amd64 linux/arm64)
serial=false
targets=()
while (($#)); do
  case "$1" in
    --serial) serial=true; shift ;;
    --platform) platforms=("${2:?--platform needs linux/amd64 or linux/arm64}"); shift 2 ;;
    --log-dir) log_dir=${2:?--log-dir needs a directory}; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    -*) echo "error: unknown option $1" >&2; usage >&2; exit 2 ;;
    *) targets+=("$1"); shift ;;
  esac
done
((${#targets[@]})) || targets=(nemo-skills nemo-rl nemo-gym vllm-0-29-0 sglang-0-5-17 nvflow-client)
for platform in "${platforms[@]}"; do
  [[ $platform == linux/amd64 || $platform == linux/arm64 ]] || fail "--platform must be linux/amd64 or linux/arm64"
done

[[ -n "${REGISTRY:-}" ]] ||
  fail "set REGISTRY to your registry and namespace, for example REGISTRY=registry.example.com/team or REGISTRY=localhost:5000/nvflow"
[[ -d .git ]] || fail "build from a normal git clone; nvflow-client does not support linked worktrees"
[[ -z "$(git status --porcelain)" ]] || fail "commit or stash local changes first; nvflow-client records HEAD"
buildx_version=$(docker buildx version)
grep -qE 'v(0\.(2[89]|[3-9][0-9])|[1-9][0-9]*)\.' <<<"$buildx_version" ||
  fail "Buildx 0.28.0 or newer is required: $buildx_version"

mkdir -p "$log_dir"
log_dir=$(cd "$log_dir" && pwd)
[[ "$log_dir/" != "$repo_root/"* ]] || fail "--log-dir must be outside the clone"

registry_host=${REGISTRY%%/*}
local_registry=false
if [[ $registry_host == localhost || $registry_host == localhost:* || $registry_host == 127.* ]]; then
  local_registry=true
fi
if ! docker buildx inspect "$builder" >/dev/null 2>&1; then
  create_args=(--name "$builder" --driver docker-container)
  if $local_registry; then
    printf '[registry."%s"]\n  http = true\n' "$registry_host" >"$log_dir/buildkitd.toml"
    create_args+=(--driver-opt network=host --buildkitd-config "$log_dir/buildkitd.toml")
  fi
  docker buildx create "${create_args[@]}" >/dev/null
fi
builder_info=$(docker buildx inspect --bootstrap "$builder")
if $local_registry && ! { grep -q 'network="host"' <<<"$builder_info" && grep -qF "\"$registry_host\"" <<<"$builder_info"; }; then
  fail "builder $builder cannot push to $registry_host: it needs host networking and plain HTTP for that registry.
Remove it with 'docker buildx rm $builder' and rerun; the script recreates it."
fi
available=$(sed -n 's/^Platforms: *//p' <<<"$builder_info" | tr -d ' *' | paste -sd, -)
for platform in "${platforms[@]}"; do
  [[ ",$available," == *",$platform,"* ]] || fail "builder $builder lacks $platform (has $available).
To emulate arm64 on an amd64 host, register QEMU and restart the builder, then rerun:
  docker run --privileged --rm tonistiigi/binfmt --install all
  docker buildx stop $builder"
done

export REGISTRY
NVFLOW_TAG="${NVFLOW_TAG:-v1.2.0}"
export NVFLOW_TAG
bake_args=(--builder "$builder" --push --provenance=false --sbom=false --progress=plain)

images=()
for target in "${targets[@]}"; do
  definition=$(docker buildx bake "$target" --print) || fail "cannot resolve Bake target $target"
  [[ $(grep -c '"tags": \[' <<<"$definition") == 1 ]] ||
    fail "$target must be a single Bake target with one image, not a group"
  images+=("$(awk '/"tags": \[/ { getline; gsub(/[ ",]/, ""); print; exit }' <<<"$definition")")
done

{
  echo "commit $(git rev-parse HEAD)"
  echo "NVFLOW_TAG $NVFLOW_TAG"
  echo "targets ${targets[*]}"
  echo "platforms ${platforms[*]}"
  echo "serial $serial"
  echo "$builder_info"
} >"$log_dir/build-info.txt"
echo "Building ${targets[*]} for ${platforms[*]}; logs in $log_dir"

bake() { # bake <target> <log-name> [extra bake args...]
  local target=$1 name=$2
  shift 2
  echo "    building $name; follow with: tail -f $log_dir/$name.log"
  if ! docker buildx bake "$target" "${bake_args[@]}" "$@" \
    --metadata-file "$log_dir/$name.metadata.json" >"$log_dir/$name.log" 2>&1; then
    tail -n 40 "$log_dir/$name.log" >&2
    fail "$name failed after $(((SECONDS - start) / 60)) min; full log: $log_dir/$name.log"
  fi
}

for i in "${!targets[@]}"; do
  target=${targets[$i]}
  image=${images[$i]}
  start=$SECONDS
  echo "==> $target started $(date '+%F %T')"
  if $serial || ((${#platforms[@]} == 1)); then
    parts=()
    for platform in "${platforms[@]}"; do
      bake "$target" "$target-${platform#linux/}" \
        --set "*.platform=$platform" --set "$target.tags=$image-${platform#linux/}"
      parts+=("$image-${platform#linux/}")
    done
    if ((${#parts[@]} == 1)); then
      image=${parts[0]}
    else
      docker buildx imagetools create --builder "$builder" --tag "$image" "${parts[@]}" \
        >"$log_dir/$target-join.log" 2>&1 || fail "could not join ${parts[*]}; see $log_dir/$target-join.log"
    fi
  else
    bake "$target" "$target"
  fi
  if ((${#platforms[@]} == 1)); then
    pushed=$(docker buildx imagetools inspect --builder "$builder" \
      --format '{{.Image.OS}}/{{.Image.Architecture}}' "$image")
    [[ $pushed == "${platforms[0]}" ]] || fail "$image is $pushed, not ${platforms[0]}"
  else
    manifest=$(docker buildx imagetools inspect --builder "$builder" "$image")
    for platform in "${platforms[@]}"; do
      grep -qE "Platform: +$platform\$" <<<"$manifest" || fail "$image is missing $platform"
    done
  fi
  echo "    pushed $image for ${platforms[*]} in $(((SECONDS - start) / 60)) min"
done
echo "All images pushed. Logs: $log_dir"
