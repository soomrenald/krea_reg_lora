from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TEXT_ENCODER_HINTS = ("clip", "text_encoder", "text.encoder", "cond_stage_model", "llm")
ATTN_HINTS = ("attn", "attention")
ATTN_OUT_HINTS = ("wo", "to_out", "out_proj", "o_proj", "proj")
MLP_HINTS = ("mlp", "ffn", "feed_forward", "feedforward", "fc1", "fc2", "up_proj", "down_proj", "gate_proj")


def is_text_encoder_lora_key(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in TEXT_ENCODER_HINTS)


def is_attention_lora_key(key: str) -> bool:
    lowered = key.lower().replace("/", ".")
    return any(hint in lowered for hint in ATTN_HINTS)


def is_writeback_lora_key(key: str) -> bool:
    lowered = key.lower().replace("/", ".")
    return is_attention_lora_key(lowered) and any(f".{hint}." in lowered or lowered.endswith(f".{hint}.weight") for hint in ATTN_OUT_HINTS) or any(
        f".{hint}." in lowered or lowered.endswith(f".{hint}.weight") for hint in MLP_HINTS
    )


def filter_lora_state_dict(
    state_dict: Mapping[str, Any],
    *,
    ignore_text_encoder_lora: bool = True,
) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key, value in state_dict.items():
        if ignore_text_encoder_lora and is_text_encoder_lora_key(key):
            continue
        filtered[key] = value
    return filtered
