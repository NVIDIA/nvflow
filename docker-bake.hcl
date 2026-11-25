variable "REGISTRY" {
}

variable "NVFLOW_TAG" {
  default = "v1.2.0"
}

variable "VLLM_0290_COMMIT" {
  default = "98dff2a81d747d1dba01a47f939f48c3526d4206"
}

variable "PYTHON_VERSION" {
  default = "3.13.14"
}

variable "RAY_VERSION" {
  default = "2.58.0"
}

variable "CUPY_VERSION" {
  default = "14.0.1"
}

variable "NEMO_GYM_IMAGE" {
  default = "nvcr.io/nvidia/nemo-gym:v0.7.0"
}

variable "NEMO_RL_IMAGE" {
  default = "nvcr.io/nvidia/nemo-rl:v0.8.0"
}

group "v1_2" {
  targets = ["nemo-skills", "nemo-rl", "nemo-gym", "sglang-0-5-17", "vllm-0-29-0", "nvflow-client"]
}

group "workers" {
  targets = ["nemo-skills", "nemo-rl", "nemo-gym", "sglang-0-5-17", "vllm-0-29-0"]
}

group "vllm" {
  targets = ["vllm-0-29-0"]
}

target "_multiarch" {
  platforms = ["linux/amd64", "linux/arm64"]
}

target "nemo-skills" {
  inherits   = ["_multiarch"]
  context    = "."
  dockerfile = "dockerfiles/Dockerfile.nemo-skills"
  tags       = ["${REGISTRY}/nvflow-nemo-skills:v1.2.0"]
  no-cache   = true
}

# The vLLM target installs vLLM's CUDA 13 wheel for the pinned commit; the
# SHA-256 values are for https://wheels.vllm.ai/<commit>/vllm-<version>-cp38-abi3-manylinux_2_28_<arch>.whl.
target "vllm-0-29-0" {
  inherits   = ["_multiarch"]
  context    = "."
  dockerfile = "dockerfiles/Dockerfile.vllm"
  args = {
    VLLM_VERSION            = "0.29.0"
    VLLM_COMMIT             = VLLM_0290_COMMIT
    VLLM_WHEEL_SHA256_AMD64 = "09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635"
    VLLM_WHEEL_SHA256_ARM64 = "e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b0"
    FLASHINFER_VERSION      = "0.6.18"
    PYTHON_VERSION          = PYTHON_VERSION
    RAY_VERSION             = RAY_VERSION
    CUPY_VERSION            = CUPY_VERSION
  }
  tags = ["${REGISTRY}/nvflow-vllm:v0.29.0"]
}

target "sglang-0-5-17" {
  inherits   = ["_multiarch"]
  context    = "."
  dockerfile = "dockerfiles/Dockerfile.sglang"
  args = {
    SGLANG_VERSION = "0.5.17"
    PYTHON_VERSION = PYTHON_VERSION
    RAY_VERSION    = RAY_VERSION
  }
  tags = ["${REGISTRY}/nvflow-sglang:v0.5.17-ray2.58.0"]
}

# NeMo-Gym 0.7.0 with the Gym component environments NVFlow's workflows use
# built in, so Gym jobs install nothing at runtime.
target "nemo-gym" {
  inherits   = ["_multiarch"]
  context    = "."
  dockerfile = "dockerfiles/Dockerfile.nemo-gym"
  args = {
    NEMO_GYM_IMAGE = NEMO_GYM_IMAGE
  }
  tags = ["${REGISTRY}/nvflow-nemo-gym:v0.7.0"]
}

# NeMo-RL 0.8.0 with the Gym component environments NVFlow's GRPO workflows
# train with built in, so training installs nothing at runtime.
target "nemo-rl" {
  inherits   = ["_multiarch"]
  context    = "."
  dockerfile = "dockerfiles/Dockerfile.nemo-rl"
  args = {
    NEMO_RL_IMAGE = NEMO_RL_IMAGE
  }
  tags = ["${REGISTRY}/nvflow-nemo-rl:v0.8.0"]
}

target "nvflow-client" {
  inherits   = ["_multiarch"]
  context    = "."
  dockerfile = "dockerfiles/Dockerfile.nvflow"
  tags       = ["${REGISTRY}/nvflow-client:${NVFLOW_TAG}"]
}
