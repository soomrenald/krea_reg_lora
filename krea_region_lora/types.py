from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F


BBoxFormat = Literal["xywh", "xyxy"]
BatchMode = Literal["single", "repeat", "per_batch"]
CrossLoraMode = Literal["allow", "penalize", "block"]
LayerTargetPolicy = Literal["attn_out_mlp", "attention_only", "all_matched_linears"]
MeasurementSource = Literal["direct_delta", "hidden_state_delta"]
MEASUREMENT_SOURCE_OPTIONS = ["direct_delta", "hidden_state_delta"]
NormalizationMode = Literal["relative_norm", "minmax", "percentile", "raw"]
OverlapMode = Literal["normalize", "priority_1", "priority_last", "add"]
RetentionMode = Literal["sticky", "decay", "instant"]


@dataclass(frozen=True)
class KreaRegion:
    region_id: str
    pixel_bbox: tuple[int, int, int, int]
    normalized_bbox: tuple[float, float, float, float]
    image_size: tuple[int, int]
    feather_px: int
    img_len: int
    text_len: int | None
    pixel_mask: torch.Tensor
    latent_mask: torch.Tensor
    token_mask: torch.Tensor
    bbox_index: int = 0
    bbox_format: BBoxFormat = "xywh"
    batch_mode: BatchMode = "repeat"
    metadata: dict[str, Any] = field(default_factory=dict)

    def mask_for(self, target: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        shape = tuple(target.shape) if isinstance(target, torch.Tensor) else tuple(target)
        device = target.device if isinstance(target, torch.Tensor) else self.latent_mask.device
        dtype = target.dtype if isinstance(target, torch.Tensor) else self.latent_mask.dtype
        if len(shape) == 5:
            mask = _fit_mask_batch(self.latent_mask, shape[0])
            mask = _fit_mask_spatial(mask, shape[-2], shape[-1]).unsqueeze(2)
            if shape[2] > 1:
                mask = mask.repeat(1, 1, shape[2], 1, 1)
            return mask.to(device=device, dtype=dtype)
        if len(shape) == 4:
            mask = _fit_mask_batch(self.latent_mask, shape[0])
            return _fit_mask_spatial(mask, shape[-2], shape[-1]).to(device=device, dtype=dtype)
        if len(shape) == 3:
            mask = _fit_mask_batch(self.token_mask, shape[0])
            return mask.to(device=device, dtype=dtype)
        if len(shape) == 2:
            mask = _fit_mask_batch(self.pixel_mask, shape[0])
            return mask.to(device=device, dtype=dtype)
        raise ValueError(f"Cannot build region mask for target shape {shape}")


@dataclass(frozen=True)
class KreaRegionalLora:
    region: KreaRegion
    lora_name: str
    positive: Any = None
    negative: Any = None
    lora_strength: float = 1.0
    delta_strength: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 1.0
    enabled: bool = True
    ignore_text_encoder_lora: bool = True
    measurement_sources: tuple[MeasurementSource, ...] = ("direct_delta",)
    normalization: NormalizationMode = "relative_norm"
    percentile: float = 95.0
    threshold: float = 0.05
    retention: RetentionMode = "sticky"
    decay: float = 0.96

    def active_at(self, step_percent: float) -> bool:
        return self.enabled and self.start_percent <= step_percent <= self.end_percent


@dataclass(frozen=True)
class KreaRegionalLoraStack:
    regions: tuple[KreaRegionalLora, ...]
    overlap_mode: OverlapMode = "normalize"
    attention_isolation_strength: float = 5.0
    cross_lora_mode: CrossLoraMode = "penalize"
    cross_lora_strength: float = 3.0

    @property
    def enabled_regions(self) -> tuple[KreaRegionalLora, ...]:
        return tuple(r for r in self.regions if r.enabled and r.lora_name not in ("", "None"))


@dataclass(frozen=True)
class KreaRegionalConditioning:
    region: KreaRegion
    conditioning: Any
    text: str
    strength: float = 1.0
    outside_strength: float = 0.0
    feather_px: int | None = None


@dataclass(frozen=True)
class KreaRegionalConditioningStack:
    global_conditioning: Any
    regions: tuple[KreaRegionalConditioning, ...]


def parse_measurement_sources(value: str | tuple[str, ...] | list[str]) -> tuple[MeasurementSource, ...]:
    if isinstance(value, str):
        raw = [p.strip().lower() for p in value.replace("+", ",").split(",")]
    else:
        raw = [str(p).strip().lower() for p in value]
    out: list[MeasurementSource] = []
    for item in raw:
        if item in ("direct", "direct_delta", "delta"):
            out.append("direct_delta")
        elif item in ("hidden", "hidden_delta", "hidden_state", "hidden_state_delta", "state_delta"):
            out.append("hidden_state_delta")
    return tuple(dict.fromkeys(out)) or ("direct_delta",)


def _fit_mask_batch(mask: torch.Tensor, batch: int) -> torch.Tensor:
    if mask.shape[0] == batch:
        return mask
    if mask.shape[0] == 1:
        reps = [int(batch)] + [1] * (mask.ndim - 1)
        return mask.repeat(*reps)
    if batch == 1:
        return mask[:1]
    raise ValueError(f"Mask batch {mask.shape[0]} does not match target batch {batch}")


def _fit_mask_spatial(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == (int(height), int(width)):
        return mask
    return F.interpolate(mask, size=(int(height), int(width)), mode="area").clamp(0.0, 1.0)
