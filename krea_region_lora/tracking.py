from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .types import KreaRegionalLora, KreaRegionalLoraStack


@dataclass
class RegionalRuntimeState:
    stack: KreaRegionalLoraStack
    debug: bool = False
    flags: dict[int, torch.Tensor] = field(default_factory=dict)
    influence: dict[int, torch.Tensor] = field(default_factory=dict)
    score_sums: dict[int, float] = field(default_factory=dict)
    updates: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.flags.clear()
        self.influence.clear()
        self.score_sums.clear()
        self.updates.clear()

    def update_from_delta(self, regional: KreaRegionalLora, delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if delta.ndim < 3:
            raise ValueError(f"Expected sequence delta tensor, got {tuple(delta.shape)}")
        seq_len = int(delta.shape[-2])
        batch = int(delta.shape[0])
        img_len = int(regional.region.token_mask.shape[1])
        if img_len <= 0:
            return delta.new_zeros((batch, seq_len), dtype=torch.bool)
        img_start = max(0, seq_len - img_len)

        raw = torch.linalg.vector_norm(delta.float(), dim=-1)
        ref = torch.linalg.vector_norm(reference.float(), dim=-1).clamp_min(1.0e-6)
        score = _normalize_score(raw, ref, regional.normalization, regional.percentile)
        image_score = score[:, img_start:img_start + img_len]
        image_region = _token_mask_for(regional, batch, delta.device, score.dtype)
        image_score = image_score * image_region.squeeze(-1)
        new_flags = image_score > float(regional.threshold)

        key = id(regional)
        old_flags = self.flags.get(key)
        old_influence = self.influence.get(key)
        if old_flags is None or old_flags.shape != new_flags.shape:
            old_flags = torch.zeros_like(new_flags)
        if old_influence is None or old_influence.shape != image_score.shape:
            old_influence = torch.zeros_like(image_score)

        if regional.retention == "sticky":
            final_flags = old_flags | new_flags
            final_influence = torch.maximum(old_influence, image_score)
        elif regional.retention == "decay":
            final_influence = torch.maximum(old_influence * float(regional.decay), image_score)
            final_flags = final_influence > float(regional.threshold)
        else:
            final_influence = image_score
            final_flags = new_flags

        self.flags[key] = final_flags.detach()
        self.influence[key] = final_influence.detach()
        self.score_sums[key] = self.score_sums.get(key, 0.0) + float(image_score.mean().detach().cpu())
        self.updates[key] = self.updates.get(key, 0) + 1
        return final_flags

    def attention_override(self, func, *args, **kwargs):
        if len(args) < 4:
            return func(*args, **kwargs)
        q, k = args[0], args[1]
        heads = args[3]
        if not torch.is_tensor(q) or not torch.is_tensor(k):
            return func(*args, **kwargs)
        q_len = int(q.shape[-2])
        k_len = int(k.shape[-2])
        if q_len != k_len:
            return func(*args, **kwargs)
        batch = int(q.shape[0])
        bias = self.build_attention_bias(batch, q_len, k_len, int(heads), q.device, q.dtype)
        if bias is None:
            return func(*args, **kwargs)
        args = list(args)
        if len(args) >= 5:
            args[4] = _combine_masks(args[4], bias)
        else:
            kwargs["mask"] = _combine_masks(kwargs.get("mask"), bias)
        return func(*args, **kwargs)

    def build_attention_bias(
        self,
        batch: int,
        q_len: int,
        k_len: int,
        heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if q_len != k_len:
            return None
        img_len = _matching_img_len(self.stack.enabled_regions, q_len)
        if img_len is None:
            return None
        img_start = q_len - img_len
        image_tokens = torch.zeros((batch, q_len), dtype=torch.bool, device=device)
        image_tokens[:, img_start:] = True
        bias = torch.zeros((batch, q_len, k_len), dtype=torch.float32, device=device)
        full_flags: list[torch.Tensor] = []
        for regional in self.stack.enabled_regions:
            flags = self.flags.get(id(regional))
            if flags is None:
                full = torch.zeros((batch, q_len), dtype=torch.bool, device=device)
            else:
                flags = _fit_batch(flags.to(device=device, dtype=torch.bool), batch)
                full = torch.zeros((batch, q_len), dtype=torch.bool, device=device)
                full[:, img_start:img_start + min(img_len, flags.shape[1])] = flags[:, :img_len]
            full_flags.append(full)
            if self.stack.attention_isolation_strength > 0:
                query = image_tokens & ~full
                key = full
                bias = bias - float(self.stack.attention_isolation_strength) * (query[:, :, None] & key[:, None, :]).float()

        cross_strength = _cross_strength(self.stack)
        if cross_strength > 0 and len(full_flags) > 1:
            for i, left in enumerate(full_flags):
                for j, right in enumerate(full_flags):
                    if i != j:
                        bias = bias - cross_strength * (left[:, :, None] & right[:, None, :]).float()

        if torch.count_nonzero(bias) == 0:
            return None
        return bias[:, None, :, :].repeat(1, max(1, int(heads)), 1, 1).reshape(batch * max(1, int(heads)), q_len, k_len).to(dtype=dtype)

    def report(self) -> str:
        lines = [
            f"enabled_regions={len(self.stack.enabled_regions)}",
            f"attention_isolation_strength={self.stack.attention_isolation_strength:.3f}",
            f"cross_lora_mode={self.stack.cross_lora_mode} cross_lora_strength={self.stack.cross_lora_strength:.3f}",
        ]
        for index, regional in enumerate(self.stack.enabled_regions, start=1):
            key = id(regional)
            flags = self.flags.get(key)
            marked = int(flags.sum().detach().cpu()) if flags is not None else 0
            updates = self.updates.get(key, 0)
            mean_score = self.score_sums.get(key, 0.0) / max(1, updates)
            lines.append(f"[{index}] {regional.lora_name}: updates={updates} marked_tokens={marked} mean_score={mean_score:.5f}")
        return "\n".join(lines)


def _normalize_score(raw: torch.Tensor, reference: torch.Tensor, mode: str, percentile: float) -> torch.Tensor:
    if mode == "raw":
        return raw
    if mode == "minmax":
        flat = raw.flatten(1)
        mn = flat.min(dim=1).values[:, None]
        mx = flat.max(dim=1).values[:, None]
        return ((flat - mn) / (mx - mn).clamp_min(1.0e-6)).reshape_as(raw)
    if mode == "percentile":
        flat = raw.flatten(1)
        denom = torch.quantile(flat, max(0.0, min(100.0, float(percentile))) / 100.0, dim=1).clamp_min(1.0e-6)
        return (raw / denom[:, None]).clamp(0.0, 1.0)
    return raw / reference


def _token_mask_for(regional: KreaRegionalLora, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = regional.region.token_mask
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.repeat(batch, 1, 1)
    elif mask.shape[0] != batch:
        mask = mask[:1].repeat(batch, 1, 1)
    return mask.to(device=device, dtype=dtype)


def _combine_masks(existing: Any, bias: torch.Tensor) -> torch.Tensor:
    if existing is None:
        return bias
    if not torch.is_tensor(existing):
        return bias
    return existing.to(device=bias.device, dtype=bias.dtype) + bias


def _fit_batch(flags: torch.Tensor, batch: int) -> torch.Tensor:
    if flags.shape[0] == batch:
        return flags
    if flags.shape[0] == 1:
        return flags.repeat(batch, 1)
    return flags[:1].repeat(batch, 1)


def _matching_img_len(regions: tuple[KreaRegionalLora, ...], seq_len: int) -> int | None:
    lengths = {int(r.region.token_mask.shape[1]) for r in regions if int(r.region.token_mask.shape[1]) <= seq_len}
    if len(lengths) == 1:
        return next(iter(lengths))
    return None


def _cross_strength(stack: KreaRegionalLoraStack) -> float:
    if stack.cross_lora_mode == "allow":
        return 0.0
    if stack.cross_lora_mode == "block":
        return 10000.0
    return max(0.0, float(stack.cross_lora_strength))
