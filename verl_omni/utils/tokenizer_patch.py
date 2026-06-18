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
"""Monkey-patch verl's hf_tokenizer to auto-load chat_template from chat_template.json.

Some models (e.g., Qwen3-Omni) store chat_template in a separate file
instead of tokenizer_config.json. This patch ensures the tokenizer
has a valid chat_template by loading it from chat_template.json
before returning it to the caller.

This patch must be applied at import time, before verl's trainer
calls ``hf_tokenizer``.
"""

import functools
import json
import os
import sys


def _load_chat_template_from_file(name_or_path: str):
    """Load chat_template from chat_template.json if available."""
    if not isinstance(name_or_path, str):
        return None
    chat_template_path = os.path.join(name_or_path, "chat_template.json")
    if os.path.exists(chat_template_path):
        try:
            with open(chat_template_path) as f:
                data = json.load(f)
                return data.get("chat_template")
        except (OSError, json.JSONDecodeError):
            pass
    return None


def _apply_chat_template_patch():
    """Install the hf_tokenizer patch."""
    from verl.utils import tokenizer as _tokenizer_module

    _original_hf_tokenizer = _tokenizer_module.hf_tokenizer

    @functools.wraps(_original_hf_tokenizer)
    def _patched_hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
        tokenizer = _original_hf_tokenizer(name_or_path, correct_pad_token, correct_gemma2, **kwargs)

        # Load chat_template from chat_template.json if not already set
        if tokenizer.chat_template is None and isinstance(name_or_path, str):
            chat_template = _load_chat_template_from_file(name_or_path)
            if chat_template:
                tokenizer.chat_template = chat_template

        return tokenizer

    # Patch the tokenizer module
    _tokenizer_module.hf_tokenizer = _patched_hf_tokenizer

    # Also patch verl.utils module attributes for callers that use `from verl.utils import hf_tokenizer`
    import verl.utils

    verl.utils.hf_tokenizer = _patched_hf_tokenizer

    # Patch sys.modules entries that already imported hf_tokenizer
    for mod_name in list(sys.modules.keys()):
        if not mod_name.startswith("verl"):
            continue
        mod = sys.modules.get(mod_name)
        if (
            mod is not None
            and hasattr(mod, "hf_tokenizer")
            and mod.__dict__.get("hf_tokenizer") is _original_hf_tokenizer
        ):
            mod.hf_tokenizer = _patched_hf_tokenizer


_apply_chat_template_patch()
