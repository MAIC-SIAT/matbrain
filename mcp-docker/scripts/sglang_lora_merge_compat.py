"""Project-local compatibility shim for verl SGLang LoRA merge mode.

verl's documented SGLang LoRA path currently requires model.lora.merge=True.
In that mode the rollout server receives merged base weights, so generation
requests should not name a separate LoRA adapter. Some verl/SGLang builds still
attach the internal adapter placeholder name to every request whenever
lora_rank > 0, which makes SGLang reject the request because no adapter was
loaded. This module removes only that placeholder assignment.

The module is intentionally inert unless MATBRAIN_SGLANG_LORA_MERGE_COMPAT=1.
It is loaded through actor_rollout_ref.model.external_lib, not by editing verl.
"""

from __future__ import annotations

import os


def _patch_generate_req_input() -> None:
    if os.environ.get("MATBRAIN_SGLANG_LORA_MERGE_COMPAT") not in {"1", "true", "True", "yes", "YES"}:
        return

    from sglang.srt.managers.io_struct import GenerateReqInput

    placeholder_name = os.environ.get("MATBRAIN_SGLANG_LORA_PLACEHOLDER", "verl_actor_lora_name")
    if getattr(GenerateReqInput, "_matbrain_lora_merge_compat", False):
        return

    original_setattr = GenerateReqInput.__setattr__

    def patched_setattr(self, name, value):  # noqa: ANN001
        if name == "lora_path" and value == placeholder_name:
            return
        return original_setattr(self, name, value)

    GenerateReqInput.__setattr__ = patched_setattr
    GenerateReqInput._matbrain_lora_merge_compat = True


_patch_generate_req_input()
