#!/bin/bash
#
# setup_containers.sh
#
# Downloads and converts NeMo-Skills containers to .sqsh format in parallel using Slurm.
#
# Usage:
#   sbatch --account=<account> scripts/setup_containers.sh --config FILE [output_dir] [--platform PLATFORM] [--force]
#
# Options:
#   --config FILE           Container definitions YAML file (required)
#   output_dir              Directory to save .sqsh files (default: ./containers)
#   --platform PLATFORM     Target platform: amd64 | arm64 (default: auto-detect from host)
#   --force                 Force re-download existing containers
#
# Examples:
#   sbatch --account=llmservice_modelalignment_sft scripts/setup_containers.sh --config cluster_configs/my_containers.yaml ./containers
#   sbatch --account=llmservice_modelalignment_sft scripts/setup_containers.sh --config cluster_configs/my_containers.yaml ./containers --platform arm64
#   sbatch --account=llmservice_modelalignment_sft scripts/setup_containers.sh --config cluster_configs/my_containers.yaml ./containers --force
#
# Platform support:
#   - Some containers are multi-arch (same tag for amd64/arm64)
#   - Some containers have platform-specific tags defined in containers.yaml
#   - Containers without support for the target platform are skipped

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --time=04:00:00
#SBATCH --job-name=nemo-containers
#SBATCH --output=outputs/logs/slurm-containers-%j.out

set -e

# =============================================================================
# Configuration
# =============================================================================

# Resolve project root (works for both sbatch and direct execution)
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_PATH")")"
fi

cd "$PROJECT_ROOT" || { echo "ERROR: Could not cd to $PROJECT_ROOT"; exit 1; }

# =============================================================================
# Output Helpers
# =============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

print_header()  { echo -e "\n${BLUE}=== $1 ===${NC}\n"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}! $1${NC}"; }
print_info()    { echo -e "${BLUE}→ $1${NC}"; }

# =============================================================================
# Utility Functions
# =============================================================================

# Detect host architecture, normalized to amd64 or arm64
get_host_arch() {
    case "$(uname -m)" in
        x86_64)  echo "amd64" ;;
        aarch64) echo "arm64" ;;
        *)       echo "" ;;  # Unsupported
    esac
}

# Validate and normalize platform
validate_platform() {
    case "$1" in
        amd64|x86_64) echo "amd64" ;;
        arm64|aarch64) echo "arm64" ;;
        *)             echo "" ;;  # Invalid
    esac
}

# Convert platform to enroot arch format (uname -m style)
platform_to_enroot_arch() {
    case "$1" in
        amd64) echo "x86_64" ;;
        arm64) echo "aarch64" ;;
    esac
}

# Extract tag from image (e.g., "repo:v1.0" -> "v1.0")
get_image_tag() {
    local tag="${1##*:}"
    [[ "$tag" == "$1" ]] && echo "latest" || echo "$tag"
}


# =============================================================================
# Dependency Check
# =============================================================================

check_dependencies() {
    print_header "Checking Dependencies"

    local missing=()
    if ! command -v yq &>/dev/null; then missing+=("yq"); fi
    if ! command -v enroot &>/dev/null; then missing+=("enroot"); fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing: ${missing[*]}"
        echo "  yq: brew install yq (macOS) or https://github.com/mikefarah/yq"
        echo "  enroot: module load enroot (or contact cluster admin)"
        exit 1
    fi

    print_success "All dependencies found"
}

# =============================================================================
# Container Parsing
# =============================================================================

parse_containers() {
    print_header "Parsing Containers"

    CONTAINER_NAMES=()
    CONTAINER_IMAGES=()
    local skipped=()

    # Read each container from YAML
    local names=$(yq -r '.containers | keys | .[]' "$YAML_FILE")

    for name in $names; do
        local value_type=$(yq -r ".containers.${name} | type" "$YAML_FILE")

        # Handle both yq formats: "string" (v4) and "!!str" (older/different versions)
        if [[ "$value_type" == "string" || "$value_type" == "!!str" ]]; then
            # Multi-arch image (single string)
            CONTAINER_NAMES+=("$name")
            CONTAINER_IMAGES+=("$(yq -r ".containers.${name}" "$YAML_FILE")")
        else
            # Platform-specific (nested object)
            local image=$(yq -r ".containers.${name}.${PLATFORM} // \"\"" "$YAML_FILE")
            if [[ -n "$image" && "$image" != "null" ]]; then
                CONTAINER_NAMES+=("$name")
                CONTAINER_IMAGES+=("$image")
            else
                skipped+=("$name")
            fi
        fi
    done

    if [[ ${#CONTAINER_NAMES[@]} -eq 0 ]]; then
        print_error "No containers found for platform: $PLATFORM"
        exit 1
    fi

    print_success "Found ${#CONTAINER_NAMES[@]} containers for $PLATFORM"

    if [[ ${#skipped[@]} -gt 0 ]]; then
        print_warning "Skipped (no $PLATFORM support): ${skipped[*]}"
    fi
}

# =============================================================================
# Download
# =============================================================================

download_containers() {
    local output_dir="$1"
    local enroot_arch=$(platform_to_enroot_arch "$PLATFORM")

    print_header "Downloading Containers"
    print_info "Architecture: $PLATFORM ($enroot_arch)"

    local launched=0

    for i in "${!CONTAINER_NAMES[@]}"; do
        local name="${CONTAINER_NAMES[$i]}"
        local image="${CONTAINER_IMAGES[$i]}"
        local tag=$(get_image_tag "$image")
        local outfile="${output_dir}/${name}-${tag}.sqsh"

        if [[ -f "$outfile" && "$FORCE" == false ]]; then
            print_warning "Skipping $name (exists)"
            continue
        fi

        if [[ "$FORCE" == true && -f "$outfile" ]]; then rm -f "$outfile"; fi

        print_info "Downloading: $name"
        srun --ntasks=1 --exclusive enroot import --arch "$enroot_arch" --output "$outfile" "docker://$image" &
        launched=$((launched + 1))
    done

    if [[ $launched -gt 0 ]]; then
        print_info "Waiting for $launched downloads..."
        wait
    else
        print_info "No new containers to download"
    fi
}

# =============================================================================
# Config Snippet Generator
# =============================================================================

generate_config_snippet() {
    local output_dir=$(readlink -f "$1")

    print_header "Configuration Snippet"
    echo "Add to your cluster config after you create it: <REPO_ROOT>/cluster_configs/my_cluster.yaml"
    echo ""
    echo "containers:"

    local found=0
    for i in "${!CONTAINER_NAMES[@]}"; do
        local name="${CONTAINER_NAMES[$i]}"
        local tag=$(get_image_tag "${CONTAINER_IMAGES[$i]}")
        local sqsh_file="$output_dir/${name}-${tag}.sqsh"
        # Only include containers that actually exist
        if [[ -f "$sqsh_file" ]]; then
            echo "  $name: $sqsh_file"
            found=$((found + 1))
        fi
    done
    echo ""

    if [[ $found -lt ${#CONTAINER_NAMES[@]} ]]; then
        print_warning "Some containers are missing - re-run to download them"
    fi
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_arguments() {
    OUTPUT_DIR="./containers"
    PLATFORM=""
    FORCE=false
    CONFIG_FILE=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                [[ -z "$2" || "$2" =~ ^-- ]] && { print_error "--config requires a file path"; exit 1; }
                CONFIG_FILE="$2"; shift 2 ;;
            --platform)
                [[ -z "$2" || "$2" =~ ^-- ]] && { print_error "--platform requires a value (amd64 or arm64)"; exit 1; }
                PLATFORM="$2"; shift 2 ;;
            --force)
                FORCE=true; shift ;;
            -*)
                print_error "Unknown option: $1"; exit 1 ;;
            *)
                OUTPUT_DIR="$1"; shift ;;
        esac
    done

    # Require --config
    if [[ -z "$CONFIG_FILE" ]]; then
        print_error "--config is required."
        echo ""
        echo "  Usage: sbatch --account=<acct> scripts/setup_containers.sh --config <FILE> [output_dir]"
        echo ""
        echo "  Create a config from the template if you haven't already:"
        echo "    cp cluster_configs/containers.yaml cluster_configs/my_containers.yaml"
        echo "    # Edit my_containers.yaml with your registry paths"
        exit 1
    fi

    # Auto-detect platform if not specified
    if [[ -z "$PLATFORM" ]]; then
        PLATFORM=$(get_host_arch)
        if [[ -z "$PLATFORM" ]]; then
            print_error "Unsupported host architecture: $(uname -m). Use --platform to specify amd64 or arm64"
            exit 1
        fi
    else
        PLATFORM=$(validate_platform "$PLATFORM")
        if [[ -z "$PLATFORM" ]]; then
            print_error "Invalid platform. Supported: amd64, arm64"
            exit 1
        fi
    fi
}

# =============================================================================
# Main
# =============================================================================

mkdir -p outputs/logs

parse_arguments "$@"

YAML_FILE="$CONFIG_FILE"
[[ -f "$YAML_FILE" ]] || { print_error "$YAML_FILE not found"; exit 1; }

check_dependencies

# Warn if cross-platform download
HOST_ARCH=$(get_host_arch)
if [[ -n "$HOST_ARCH" && "$HOST_ARCH" != "$PLATFORM" ]]; then
    print_warning "Downloading $PLATFORM containers on $HOST_ARCH host (won't run locally)"
elif [[ -z "$HOST_ARCH" ]]; then
    HOST_ARCH="unknown"
fi

parse_containers
mkdir -p "$OUTPUT_DIR"

print_header "Configuration"
cat <<EOF
Config:    $YAML_FILE
Output:    $OUTPUT_DIR
Platform:  $PLATFORM (host: $HOST_ARCH)
Force:     $FORCE
Containers: ${#CONTAINER_NAMES[@]}
EOF

download_containers "$OUTPUT_DIR"

print_header "Summary"

# Check which containers exist vs. expected
FAILED_DOWNLOADS=()
for i in "${!CONTAINER_NAMES[@]}"; do
    name="${CONTAINER_NAMES[$i]}"
    tag=$(get_image_tag "${CONTAINER_IMAGES[$i]}")
    sqsh_file="$OUTPUT_DIR/${name}-${tag}.sqsh"
    if [[ ! -f "$sqsh_file" ]]; then
        FAILED_DOWNLOADS+=("$name")
    fi
done

ls -lh "$OUTPUT_DIR"/*.sqsh 2>/dev/null || print_warning "No .sqsh files found"

if [[ ${#FAILED_DOWNLOADS[@]} -gt 0 ]]; then
    echo ""
    print_error "Missing: ${FAILED_DOWNLOADS[*]}"
fi

generate_config_snippet "$OUTPUT_DIR"

if [[ ${#FAILED_DOWNLOADS[@]} -gt 0 ]]; then
    print_warning "Completed with errors - ${#FAILED_DOWNLOADS[@]} container(s) failed"
    exit 1
else
    print_success "Done!"
fi
