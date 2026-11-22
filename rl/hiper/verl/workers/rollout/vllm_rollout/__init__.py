# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

try:
    vllm_package_version = version("vllm")
except PackageNotFoundError as exc:
    raise PackageNotFoundError("To use vllm rollout, please ensure the 'vllm' package (>=0.7.0) is installed.") from exc

if Version(vllm_package_version) < Version("0.7.0"):
    raise ValueError(f"vllm>=0.7.0 is required for SPMD rollout; found {vllm_package_version!r}")

vllm_mode = "spmd"
from .vllm_rollout_spmd import vLLMRollout  # noqa: E402, F401
