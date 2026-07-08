from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F


BBoxFormat = Literal["xywh", "xyxy"]
BatchMode = Literal["single", "repeat", "per_batch"]
OverlapMode = Literal["normalize", "priority_1", "priority_3", "add_clamped"]


@dataclass(frozen=True)
class K2Region:
    pixel_bbox: tuple[int, int, int, int]
    image_size: tuple[int, int]
    pixel_mask: torch.Tensor
    latent_mask: torch.Tensor
    token_mask: torch.Tensor
    bbox_format: BBoxFormat = "xywh"
    bbox_index: int = 0
    batch_mode: BatchMode = "repeat"
    metadata: dict[str, Any] = field(default_factory=dict)

    def mask_for(self, target: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        shape = tuple(target.shape) if isinstance(target, torch.Tensor) else tuple(target)
        device = target.device if isinstance(target, torch.Tensor) else self.latent_mask.device
        dtype = target.dtype if isinstance(target, torch.Tensor) else self.latent_mask.dtype
        if len(shape) == 5:
            mask = self.latent_mask
            mask = _fit_mask_batch(mask, shape[0])
            mask = _fit_mask_spatial(mask, shape[-2], shape[-1])
            mask = mask.unsqueeze(2)
            if shape[2] > 1:
                mask = mask.repeat(1, 1, shape[2], 1, 1)
            return mask.to(device=device, dtype=dtype)
        if len(shape) == 4:
            mask = self.latent_mask
            mask = _fit_mask_batch(mask, shape[0])
            return _fit_mask_spatial(mask, shape[-2], shape[-1]).to(device=device, dtype=dtype)
        if len(shape) == 3:
            mask = self.token_mask
            return _fit_mask_batch(mask, shape[0]).to(device=device, dtype=dtype)
        if len(shape) == 2:
            mask = self.pixel_mask
            return _fit_mask_batch(mask, shape[0]).to(device=device, dtype=dtype)
        raise ValueError(f"Cannot build a region mask for shape {shape}")


@dataclass(frozen=True)
class K2RegionalLora:
    region: K2Region
    positive: Any
    negative: Any
    lora_name: str
    lora_strength: float = 1.0
    delta_strength: float = 1.0
    start_percent: float = 0.10
    end_percent: float = 0.95
    enabled: bool = True
    attention_only_filter: bool = True
    ignore_text_encoder_lora: bool = True

    def active_at(self, step_percent: float) -> bool:
        return self.enabled and self.start_percent <= step_percent <= self.end_percent


@dataclass(frozen=True)
class K2RegionalLoraStack:
    regions: tuple[K2RegionalLora, ...]
    overlap_mode: OverlapMode = "normalize"

    @property
    def enabled_regions(self) -> tuple[K2RegionalLora, ...]:
        return tuple(r for r in self.regions if r.enabled)


def _fit_mask_batch(mask: torch.Tensor, batch: int) -> torch.Tensor:
    if mask.shape[0] == batch:
        return mask
    if mask.shape[0] == 1:
        reps = [batch] + [1] * (mask.ndim - 1)
        return mask.repeat(*reps)
    if batch == 1:
        return mask[:1]
    raise ValueError(f"Mask batch {mask.shape[0]} does not match target batch {batch}")


def _fit_mask_spatial(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == (int(height), int(width)):
        return mask
    return F.interpolate(mask, size=(int(height), int(width)), mode="area").clamp(0.0, 1.0)
