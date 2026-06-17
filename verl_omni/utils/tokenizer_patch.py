# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Thin wrapper that triggers Qwen3-Omni Thinker model patches on import.

Loaded via ``VERL_USE_EXTERNAL_MODULES=verl_omni.utils.tokenizer_patch`` or
imported directly from ``verl_omni.__init__``.  Delegates to the model-specific
patch module (:mod:`verl_omni.models.transformers.qwen3_omni_thinker`).
"""

from verl_omni.models.transformers.qwen3_omni_thinker import apply_qwen3_omni_thinker_patches

apply_qwen3_omni_thinker_patches()
