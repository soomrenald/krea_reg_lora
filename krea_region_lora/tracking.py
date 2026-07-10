from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .types import KreaRegion, KreaRegionalLora, KreaRegionalLoraStack


LOGGER = logging.getLogger(__name__)
FORBID_BIAS = 10000.0


@dataclass(frozen=True)
class SequenceLayout:
    seq_len: int
    text_len: int
    img_len: int
    pad_len: int
    img_start: int
    img_end: int


@dataclass(frozen=True)
class MaskStats:
    shape: tuple[int, ...]
    nonzero_range: tuple[int, int] | None
    min: float
    max: float
    mean: float
    checksum: float


@dataclass
class RegionalRuntimeState:
    stack: KreaRegionalLoraStack
    debug: bool = False
    flags: dict[int, torch.Tensor] = field(default_factory=dict)
    influence: dict[int, torch.Tensor] = field(default_factory=dict)
    score_sums: dict[int, float] = field(default_factory=dict)
    updates: dict[int, int] = field(default_factory=dict)
    text_len: int | None = None
    img_len: int | None = None
    image_rows: int | None = None
    image_cols: int | None = None
    seq_len: int | None = None
    pad_len: int | None = None
    img_start: int | None = None
    img_end: int | None = None
    mask_ranges: dict[int, tuple[int, int] | None] = field(default_factory=dict)
    lora_mask_stats: dict[str, MaskStats] = field(default_factory=dict)
    logged_layouts: set[tuple[int, int, int, int, int, int]] = field(default_factory=set)
    logged_attention_calls: set[tuple[str, int, int, int, int, int]] = field(default_factory=set)

    def reset(self) -> None:
        self.flags.clear()
        self.influence.clear()
        self.score_sums.clear()
        self.updates.clear()
        self.text_len = None
        self.img_len = None
        self.image_rows = None
        self.image_cols = None
        self.seq_len = None
        self.pad_len = None
        self.img_start = None
        self.img_end = None
        self.mask_ranges.clear()
        self.lora_mask_stats.clear()
        self.logged_layouts.clear()
        self.logged_attention_calls.clear()

    def capture_layout(self, args: tuple[Any, ...], kwargs: dict[str, Any], model_obj: Any) -> None:
        context = kwargs.get("context")
        if context is None and len(args) > 2:
            context = args[2]
        if torch.is_tensor(context) and context.ndim >= 3:
            self.text_len = int(context.shape[1])

        latent = kwargs.get("x")
        if latent is None and args:
            latent = args[0]
        if torch.is_tensor(latent) and latent.ndim >= 4:
            patch = int(getattr(model_obj, "patch", getattr(model_obj, "patch_size", 2)) or 2)
            h = int(latent.shape[-2])
            w = int(latent.shape[-1])
            rows = max(1, math.ceil(h / patch))
            cols = max(1, math.ceil(w / patch))
            self.image_rows = rows
            self.image_cols = cols
            self.img_len = rows * cols
        elif self.img_len is None:
            lengths = {int(r.region.img_len) for r in self.stack.enabled_regions if int(r.region.img_len) > 0}
            if len(lengths) == 1:
                self.img_len = next(iter(lengths))

    def update_from_delta(self, regional: KreaRegionalLora, delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return self.update_from_measurements(regional, [(delta, reference)])

    def update_from_measurements(self, regional: KreaRegionalLora, measurements: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not measurements:
            raise ValueError("Expected at least one delta measurement")
        first_delta = measurements[0][0]
        if first_delta.ndim < 3:
            raise ValueError(f"Expected sequence delta tensor, got {tuple(first_delta.shape)}")
        seq_len = int(first_delta.shape[-2])
        batch = int(first_delta.shape[0])
        layout = self.layout_for(regional, seq_len)
        img_len = int(regional.region.img_len)
        if layout is None:
            return first_delta.new_zeros((batch, img_len), dtype=torch.bool)

        scores = []
        for delta, reference in measurements:
            if delta.ndim < 3:
                raise ValueError(f"Expected sequence delta tensor, got {tuple(delta.shape)}")
            if tuple(delta.shape[:-1]) != tuple(first_delta.shape[:-1]):
                raise ValueError(f"Measurement sequence shape {tuple(delta.shape)} does not match {tuple(first_delta.shape)}")
            raw = torch.linalg.vector_norm(delta.float(), dim=-1)
            ref = torch.linalg.vector_norm(reference.float(), dim=-1).clamp_min(1.0e-6)
            scores.append(_normalize_score(raw, ref, regional.normalization, regional.percentile))
        score = torch.stack(scores, dim=0).mean(dim=0)
        image_score = score[:, layout.img_start:layout.img_end]
        image_region = token_mask_for_region(regional.region, batch, first_delta.device, score.dtype, layout.img_len, self.image_rows, self.image_cols)
        self.lora_mask_stats[regional.region.region_id] = mask_stats(image_region.squeeze(-1), offset=layout.img_start)
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
        self._record_mask_range(key, final_flags, layout.img_start)
        if self.debug:
            stats = self.lora_mask_stats[regional.region.region_id]
            LOGGER.info(
                "[KreaRegionalLoRA] region_id=%s seq_len=%d text_len=%d img_len=%d pad_len=%d img_start=%d img_end=%d "
                "modified_nonzero_range=%s lora_mask_range=%s mask_shape=%s mask_min=%.4f mask_max=%.4f mask_mean=%.4f mask_checksum=%.4f",
                regional.region.region_id,
                layout.seq_len,
                layout.text_len,
                layout.img_len,
                layout.pad_len,
                layout.img_start,
                layout.img_end,
                self.mask_ranges.get(key),
                stats.nonzero_range,
                stats.shape,
                stats.min,
                stats.max,
                stats.mean,
                stats.checksum,
            )
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
        backend = attention_backend_name(func)
        bias = self.build_attention_bias(batch, q_len, k_len, int(heads), q.device, q.dtype)
        if bias is None:
            return func(*args, **kwargs)
        if not additive_bias_supported(func):
            raise RuntimeError(f"Active attention backend/path {backend} cannot accept additive attention bias")
        if self.debug:
            layout = self.layout_for(self.stack.enabled_regions[0], q_len) if self.stack.enabled_regions else None
            if layout is not None:
                key = (backend, layout.seq_len, layout.text_len, layout.img_len, layout.img_start, layout.img_end)
                if key not in self.logged_attention_calls:
                    self.logged_attention_calls.add(key)
                    LOGGER.info(
                        "[KreaRegionalLoRA] attention class=%s function=%s hook_boundary=%s hidden_shape=%s attention_shape=%s "
                        "text_len=%d image_len=%d pad_len=%d image_start=%d image_end=%d backend=%s",
                        "comfy.ldm.krea2.model.Attention",
                        "Attention.forward",
                        "optimized_attention_masked mask argument",
                        tuple(q.shape),
                        tuple(bias.shape),
                        layout.text_len,
                        layout.img_len,
                        layout.pad_len,
                        layout.img_start,
                        layout.img_end,
                        backend,
                    )
        args = list(args)
        if len(args) >= 5:
            args[4] = combine_attention_masks(args[4], bias)
        else:
            kwargs["mask"] = combine_attention_masks(kwargs.get("mask"), bias)
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
        regional_layouts = [(regional, self.layout_for(regional, q_len)) for regional in self.stack.enabled_regions]
        regional_layouts = [(regional, layout) for regional, layout in regional_layouts if layout is not None]
        if not regional_layouts:
            return None
        first_layout = regional_layouts[0][1]
        assert first_layout is not None
        image_tokens = torch.zeros((batch, q_len), dtype=torch.bool, device=device)
        image_tokens[:, first_layout.img_start:first_layout.img_end] = True
        bias = torch.zeros((batch, q_len, k_len), dtype=torch.float32, device=device)
        full_flags: list[torch.Tensor] = []
        for regional, layout in regional_layouts:
            assert layout is not None
            flags = self.flags.get(id(regional))
            full = torch.zeros((batch, q_len), dtype=torch.bool, device=device)
            if flags is not None:
                flags = _fit_batch(flags.to(device=device, dtype=torch.bool), batch)
                full[:, layout.img_start:layout.img_end] = flags[:, :layout.img_len]
            full_flags.append(full)
            inward_strength = _mode_strength(self.stack.attention_isolation_mode, self.stack.attention_isolation_strength)
            if inward_strength > 0:
                query = image_tokens & ~full
                key = full
                bias = bias - inward_strength * (query[:, :, None] & key[:, None, :]).float()
            outward_strength = _mode_strength(self.stack.modified_outward_mode, self.stack.modified_outward_strength)
            if outward_strength > 0:
                query = full
                key = image_tokens & ~full
                bias = bias - outward_strength * (query[:, :, None] & key[:, None, :]).float()

        cross_strength = _cross_strength(self.stack)
        if cross_strength > 0 and len(full_flags) > 1:
            for i, left in enumerate(full_flags):
                for j, right in enumerate(full_flags):
                    if i != j:
                        bias = bias - cross_strength * (left[:, :, None] & right[:, None, :]).float()

        if torch.count_nonzero(bias) == 0:
            return None
        return expand_bias_to_heads(bias, heads).to(dtype=dtype)

    def layout_for(self, regional: KreaRegionalLora, seq_len: int) -> SequenceLayout | None:
        img_len = int(self.img_len or regional.region.img_len)
        if img_len <= 0 or img_len > seq_len:
            return None
        if self.text_len is not None:
            text_len = int(self.text_len)
        elif regional.region.text_len is not None:
            text_len = int(regional.region.text_len)
        elif seq_len == img_len:
            text_len = 0
        else:
            return None
        img_start = text_len
        img_end = img_start + img_len
        if img_start < 0 or img_end > seq_len:
            return None
        layout = SequenceLayout(seq_len=seq_len, text_len=text_len, img_len=img_len, pad_len=seq_len - img_end, img_start=img_start, img_end=img_end)
        self.seq_len = layout.seq_len
        self.text_len = layout.text_len
        self.img_len = layout.img_len
        self.pad_len = layout.pad_len
        self.img_start = layout.img_start
        self.img_end = layout.img_end
        layout_key = (layout.seq_len, layout.text_len, layout.img_len, layout.pad_len, layout.img_start, layout.img_end)
        if self.debug and layout_key not in self.logged_layouts:
            self.logged_layouts.add(layout_key)
            LOGGER.info(
                "[KreaRegionalLoRA] token layout seq_len=%d text_len=%d img_len=%d pad_len=%d img_start=%d img_end=%d",
                layout.seq_len,
                layout.text_len,
                layout.img_len,
                layout.pad_len,
                layout.img_start,
                layout.img_end,
            )
        return layout

    def report(self) -> str:
        lines = [
            f"enabled_regions={len(self.stack.enabled_regions)}",
            f"attention_isolation_mode={self.stack.attention_isolation_mode} attention_isolation_strength={self.stack.attention_isolation_strength:.3f}",
            f"modified_outward_mode={self.stack.modified_outward_mode} modified_outward_strength={self.stack.modified_outward_strength:.3f}",
            f"cross_lora_mode={self.stack.cross_lora_mode} cross_lora_strength={self.stack.cross_lora_strength:.3f}",
            f"layout seq_len={self.seq_len} text_len={self.text_len} img_len={self.img_len} pad_len={self.pad_len} img_start={self.img_start} img_end={self.img_end}",
        ]
        for index, regional in enumerate(self.stack.enabled_regions, start=1):
            key = id(regional)
            flags = self.flags.get(key)
            marked = int(flags.sum().detach().cpu()) if flags is not None else 0
            updates = self.updates.get(key, 0)
            mean_score = self.score_sums.get(key, 0.0) / max(1, updates)
            lines.append(
                f"[{index}] {regional.lora_name}: updates={updates} marked_tokens={marked} "
                f"mean_score={mean_score:.5f} nonzero_mask_range={self.mask_ranges.get(key)}"
            )
        return "\n".join(lines)

    def _record_mask_range(self, key: int, flags: torch.Tensor, img_start: int) -> None:
        if flags.numel() == 0 or torch.count_nonzero(flags) == 0:
            self.mask_ranges[key] = None
            return
        positions = torch.nonzero(flags, as_tuple=False)[:, -1]
        self.mask_ranges[key] = (img_start + int(positions.min().detach().cpu()), img_start + int(positions.max().detach().cpu()))


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


def token_mask_for_region(
    region: KreaRegion,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    img_len: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
) -> torch.Tensor:
    target_len = int(img_len or region.img_len)
    token_mask = region.token_mask
    if int(token_mask.shape[1]) != target_len:
        token_mask = resize_region_token_mask(region, target_len, rows, cols)
    if token_mask.shape[0] == 1 and batch > 1:
        token_mask = token_mask.repeat(batch, 1, 1)
    elif token_mask.shape[0] != batch:
        token_mask = token_mask[:1].repeat(batch, 1, 1)
    return token_mask.to(device=device, dtype=dtype)


def resize_region_token_mask(region: KreaRegion, img_len: int, rows: int | None, cols: int | None) -> torch.Tensor:
    if rows is not None and cols is not None and int(rows) * int(cols) == int(img_len):
        return F.interpolate(region.pixel_mask.unsqueeze(1), size=(int(rows), int(cols)), mode="area").clamp(0.0, 1.0).flatten(2).transpose(1, 2)
    mask = region.token_mask.transpose(1, 2)
    return F.interpolate(mask, size=int(img_len), mode="linear", align_corners=False).clamp(0.0, 1.0).transpose(1, 2)


def mask_stats(mask: torch.Tensor, *, offset: int = 0) -> MaskStats:
    m = mask.detach().float()
    if m.numel() == 0:
        return MaskStats(tuple(mask.shape), None, 0.0, 0.0, 0.0, 0.0)
    nonzero = torch.nonzero(m > 0, as_tuple=False)
    if nonzero.numel() == 0:
        nonzero_range = None
    else:
        positions = nonzero[:, -1]
        nonzero_range = (offset + int(positions.min().cpu()), offset + int(positions.max().cpu()))
    return MaskStats(
        tuple(mask.shape),
        nonzero_range,
        float(m.min().cpu()),
        float(m.max().cpu()),
        float(m.mean().cpu()),
        float(m.sum().cpu()),
    )


def combine_attention_masks(existing: Any, bias: torch.Tensor) -> torch.Tensor:
    if existing is None:
        return bias
    if not torch.is_tensor(existing):
        return bias
    return existing.to(device=bias.device, dtype=bias.dtype) + bias


def expand_bias_to_heads(bias: torch.Tensor, heads: int) -> torch.Tensor:
    batch, q_len, k_len = bias.shape
    h = max(1, int(heads))
    return bias[:, None, :, :].repeat(1, h, 1, 1).reshape(batch * h, q_len, k_len)


def attention_backend_name(func: Any) -> str:
    module = getattr(func, "__module__", "")
    name = getattr(func, "__name__", type(func).__name__)
    return f"{module}.{name}" if module else str(name)


def additive_bias_supported(func: Any) -> bool:
    name = getattr(func, "__name__", "")
    if name in {"attention_flash", "attention3_sage"}:
        return False
    return True


def _fit_batch(flags: torch.Tensor, batch: int) -> torch.Tensor:
    if flags.shape[0] == batch:
        return flags
    if flags.shape[0] == 1:
        return flags.repeat(batch, 1)
    return flags[:1].repeat(batch, 1)


def _mode_strength(mode: str, strength: float) -> float:
    if mode == "none":
        return 0.0
    if mode == "forbid":
        return FORBID_BIAS
    return max(0.0, float(strength))


def _cross_strength(stack: KreaRegionalLoraStack) -> float:
    if stack.cross_lora_mode == "allow":
        return 0.0
    if stack.cross_lora_mode == "block":
        return FORBID_BIAS
    return max(0.0, float(stack.cross_lora_strength))
