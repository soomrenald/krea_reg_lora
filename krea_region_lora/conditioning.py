from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .masks import make_pixel_mask
from .tracking import (
    FORBID_BIAS,
    additive_bias_supported,
    attention_backend_name,
    combine_attention_masks,
    expand_bias_to_heads,
    mask_stats,
    token_mask_for_region,
)
from .types import ConditioningMode, KreaRegionalConditioning, KreaRegionalConditioningStack, PromptAttentionMode


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptTokenRange:
    region_id: str
    start: int
    end: int
    regional: KreaRegionalConditioning


@dataclass
class RegionalConditioningReport:
    lines: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class RegionalPromptRuntimeState:
    stack: KreaRegionalConditioningStack
    debug: bool = False
    global_text_len: int | None = None
    text_len: int | None = None
    img_len: int | None = None
    image_rows: int | None = None
    image_cols: int | None = None
    seq_len: int | None = None
    pad_len: int | None = None
    image_start: int | None = None
    image_end: int | None = None
    prompt_ranges: list[PromptTokenRange] = field(default_factory=list)
    prompt_mask_stats: dict[str, Any] = field(default_factory=dict)
    prompt_attention_ranges: dict[str, tuple[tuple[int, int] | None, tuple[int, int] | None]] = field(default_factory=dict)
    logged_layouts: set[tuple[int, int, int, int, int, int]] = field(default_factory=set)
    logged_attention_calls: set[tuple[str, int, int, int, int, int]] = field(default_factory=set)

    def reset(self) -> None:
        self.global_text_len = None
        self.text_len = None
        self.img_len = None
        self.image_rows = None
        self.image_cols = None
        self.seq_len = None
        self.pad_len = None
        self.image_start = None
        self.image_end = None
        self.prompt_ranges.clear()
        self.prompt_mask_stats.clear()
        self.prompt_attention_ranges.clear()
        self.logged_layouts.clear()
        self.logged_attention_calls.clear()

    def prepare_forward(self, args: tuple[Any, ...], kwargs: dict[str, Any], model_obj: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args_list = list(args)
        context, source = _extract_context(args_list, kwargs)
        if not torch.is_tensor(context) or context.ndim != 3:
            self.reset()
            raise RuntimeError("Regional prompt attention bias requires a real Krea2 context tensor with token positions")
        original_text_len = int(context.shape[1])
        self.global_text_len = original_text_len
        pieces = [context]
        ranges: list[PromptTokenRange] = []
        cursor = original_text_len
        for regional in self.stack.regions:
            regional_context = _conditioning_tensor(regional.conditioning)
            if regional_context is None:
                raise RuntimeError(f"Regional prompt {regional.region.region_id} has no tensor conditioning; cannot identify real prompt-token key positions")
            regional_context = _fit_context_tensor(regional_context, context)
            start = cursor
            end = start + int(regional_context.shape[1])
            pieces.append(regional_context)
            ranges.append(PromptTokenRange(regional.region.region_id, start, end, regional))
            cursor = end
        if len(pieces) == 1:
            self.reset()
            return tuple(args_list), kwargs
        augmented = torch.cat(pieces, dim=1)
        _replace_context(args_list, kwargs, source, augmented)
        self.prompt_ranges = ranges
        self.text_len = int(augmented.shape[1])
        self._capture_image_layout(args_list, kwargs, model_obj)
        if self.debug:
            LOGGER.info(
                "[KreaRegionalPrompt] appended regional prompt tokens global_text_len=%d text_len=%d ranges=%s",
                original_text_len,
                self.text_len,
                [(r.region_id, r.start, r.end) for r in ranges],
            )
        return tuple(args_list), kwargs

    def _capture_image_layout(self, args: list[Any], kwargs: dict[str, Any], model_obj: Any) -> None:
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
            lengths = {int(r.region.img_len) for r in self.stack.regions if int(r.region.img_len) > 0}
            if len(lengths) == 1:
                self.img_len = next(iter(lengths))

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
        bias = self.build_prompt_attention_bias(batch, q_len, k_len, int(heads), q.device, q.dtype)
        if bias is None:
            return func(*args, **kwargs)
        backend = attention_backend_name(func)
        if not additive_bias_supported(func):
            raise RuntimeError(f"Active attention backend/path {backend} cannot accept additive attention bias")
        if self.debug:
            key = (backend, q_len, int(self.text_len or 0), int(self.img_len or 0), int(self.image_start or 0), int(self.image_end or 0))
            if key not in self.logged_attention_calls:
                self.logged_attention_calls.add(key)
                LOGGER.info(
                    "[KreaRegionalPrompt] attention class=%s function=%s hook_boundary=%s hidden_shape=%s attention_shape=%s "
                    "text_len=%d image_len=%d pad_len=%d image_start=%d image_end=%d backend=%s prompt_ranges=%s bias_nonzero=%d",
                    "comfy.ldm.krea2.model.Attention",
                    "Attention.forward",
                    "optimized_attention_masked mask argument",
                    tuple(q.shape),
                    tuple(bias.shape),
                    int(self.text_len or 0),
                    int(self.img_len or 0),
                    int(self.pad_len or 0),
                    int(self.image_start or 0),
                    int(self.image_end or 0),
                    backend,
                    [(r.region_id, r.start, r.end) for r in self.prompt_ranges],
                    int(torch.count_nonzero(bias).detach().cpu()),
                )
        args = list(args)
        if len(args) >= 5:
            args[4] = combine_attention_masks(args[4], bias)
        else:
            kwargs["mask"] = combine_attention_masks(kwargs.get("mask"), bias)
        return func(*args, **kwargs)

    def build_prompt_attention_bias(
        self,
        batch: int,
        q_len: int,
        k_len: int,
        heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if q_len != k_len or self.text_len is None or self.img_len is None:
            return None
        text_len = int(self.text_len)
        img_len = int(self.img_len)
        image_start = text_len
        image_end = image_start + img_len
        if image_start < 0 or image_end > q_len:
            return None
        self.seq_len = q_len
        self.pad_len = q_len - image_end
        self.image_start = image_start
        self.image_end = image_end
        layout_key = (q_len, text_len, img_len, self.pad_len, image_start, image_end)
        if self.debug and layout_key not in self.logged_layouts:
            self.logged_layouts.add(layout_key)
            LOGGER.info(
                "[KreaRegionalPrompt] token layout seq_len=%d text_len=%d img_len=%d pad_len=%d image_start=%d image_end=%d",
                q_len,
                text_len,
                img_len,
                self.pad_len,
                image_start,
                image_end,
            )
        image_tokens = torch.zeros((batch, q_len), dtype=torch.bool, device=device)
        image_tokens[:, image_start:image_end] = True
        bias = torch.zeros((batch, q_len, k_len), dtype=torch.float32, device=device)
        for prompt_range in self.prompt_ranges:
            regional = prompt_range.regional
            if prompt_range.start < 0 or prompt_range.end > text_len or prompt_range.start >= prompt_range.end:
                raise RuntimeError(f"Invalid prompt token range for region {prompt_range.region_id}: {prompt_range.start}:{prompt_range.end}")
            token_mask = token_mask_for_region(regional.region, batch, device, torch.float32, img_len, self.image_rows, self.image_cols).squeeze(-1)
            self.prompt_mask_stats[prompt_range.region_id] = mask_stats(token_mask, offset=image_start)
            if torch.count_nonzero(token_mask) == 0:
                self.prompt_attention_ranges[prompt_range.region_id] = (None, (prompt_range.start, prompt_range.end - 1))
                continue
            mode_strength = _prompt_mode_strength(regional.prompt_attention_mode, regional.prompt_attention_strength, regional.outside_strength)
            if mode_strength <= 0.0:
                continue
            in_region_image = torch.zeros((batch, q_len), dtype=torch.bool, device=device)
            in_region_image[:, image_start:image_end] = token_mask > 0
            outside_image = image_tokens & ~in_region_image
            prompt_keys = torch.zeros((batch, k_len), dtype=torch.bool, device=device)
            prompt_keys[:, prompt_range.start:prompt_range.end] = True
            bias = bias - mode_strength * (outside_image[:, :, None] & prompt_keys[:, None, :]).float()
            query_range = mask_stats(outside_image[:, image_start:image_end].float(), offset=image_start).nonzero_range
            key_range = (prompt_range.start, prompt_range.end - 1)
            self.prompt_attention_ranges[prompt_range.region_id] = (query_range, key_range)
            if self.debug:
                stats = self.prompt_mask_stats[prompt_range.region_id]
                LOGGER.info(
                    "[KreaRegionalPrompt] region_id=%s prompt_region_id=%s image_token_mask_shape=%s "
                    "prompt_attention_query_range=%s prompt_attention_key_range=%s mask_min=%.4f mask_max=%.4f mask_mean=%.4f mask_checksum=%.4f",
                    regional.region.region_id,
                    prompt_range.region_id,
                    stats.shape,
                    query_range,
                    key_range,
                    stats.min,
                    stats.max,
                    stats.mean,
                    stats.checksum,
                )
        if torch.count_nonzero(bias) == 0:
            return None
        return expand_bias_to_heads(bias, heads).to(dtype=dtype)


def encode_regional_conditioning(
    clip: Any,
    region: Any,
    text: str,
    strength: float,
    outside_strength: float,
    feather: int,
    prompt_attention_mode: PromptAttentionMode = "forbid",
    prompt_attention_strength: float = 5.0,
) -> KreaRegionalConditioning:
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None")
    tokens = clip.tokenize(text)
    conditioning = _attach_region_metadata(
        clip.encode_from_tokens_scheduled(tokens),
        region,
        float(strength),
        float(outside_strength),
        None if int(feather) < 0 else int(feather),
        prompt_attention_mode,
        float(prompt_attention_strength),
    )
    return KreaRegionalConditioning(
        region=region,
        conditioning=conditioning,
        text=str(text),
        region_id=region.region_id,
        token_range_metadata=None,
        strength=float(strength),
        outside_strength=float(outside_strength),
        prompt_attention_mode=prompt_attention_mode,
        prompt_attention_strength=float(prompt_attention_strength),
        feather_px=None if int(feather) < 0 else int(feather),
    )


def build_conditioning_stack(global_conditioning: Any, regions: list[KreaRegionalConditioning]) -> KreaRegionalConditioningStack:
    return KreaRegionalConditioningStack(global_conditioning=global_conditioning, regions=tuple(regions))


def build_regional_conditioning_model(
    model: Any,
    stack: KreaRegionalConditioningStack,
    *,
    conditioning_mode: ConditioningMode = "transformer_attention_bias",
    debug: bool = False,
) -> tuple[Any, str]:
    if not stack.regions:
        return model, "No regional conditioning entries; model unchanged."
    if conditioning_mode == "disabled":
        return model, "Regional prompt conditioning disabled; model unchanged."
    if conditioning_mode == "sampler_delta_conditioning":
        return _build_sampler_delta_conditioning_model(model, stack, debug=debug)
    if conditioning_mode != "transformer_attention_bias":
        raise RuntimeError(f"Unsupported regional conditioning mode: {conditioning_mode}")
    return _build_transformer_attention_bias_model(model, stack, debug=debug)


def _build_transformer_attention_bias_model(model: Any, stack: KreaRegionalConditioningStack, *, debug: bool) -> tuple[Any, str]:
    try:
        import comfy.patcher_extension  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised inside ComfyUI
        raise RuntimeError("ComfyUI patcher APIs are not importable") from exc

    for regional in stack.regions:
        if _conditioning_tensor(regional.conditioning) is None:
            raise RuntimeError(f"Regional prompt {regional.region.region_id} has no tensor conditioning; cannot identify real prompt-token key positions")
    model_out = model.clone()
    state = RegionalPromptRuntimeState(stack=stack, debug=debug)
    state.reset()

    def wrapper(executor, *args, **kwargs):
        model_obj = getattr(executor, "class_obj", None)
        if model_obj is None:
            return executor(*args, **kwargs)
        new_args, new_kwargs = state.prepare_forward(args, dict(kwargs), model_obj)
        return executor(*new_args, **new_kwargs)

    model_out.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "krea_regional_prompt_attention_bias",
        wrapper,
    )
    prepare_sampling = getattr(comfy.patcher_extension.WrappersMP, "PREPARE_SAMPLING", None)
    if prepare_sampling is not None:
        model_out.add_wrapper_with_key(prepare_sampling, "krea_regional_prompt_reset", _make_reset_wrapper(state))
    _install_prompt_attention_override(model_out, state)

    report = RegionalConditioningReport()
    report.add(f"Regional prompt conditioning: mode=transformer_attention_bias regions={len(stack.regions)}")
    report.add("Hook target: class=comfy.ldm.krea2.model.Attention function=Attention.forward boundary=optimized_attention_masked mask argument")
    report.add("Regional prompt tokens are appended to the runtime Krea2 text token sequence and preserved as prompt token ranges.")
    for index, regional in enumerate(stack.regions, start=1):
        report.add(_region_report_line(index, regional, "transformer_attention_bias"))
    return model_out, report.text()


def _build_sampler_delta_conditioning_model(model: Any, stack: KreaRegionalConditioningStack, *, debug: bool = False) -> tuple[Any, str]:
    model_out = model.clone()
    previous = getattr(model_out, "model_options", {}).get("sampler_cfg_function")
    report = RegionalConditioningReport()
    report.add(f"Regional prompt conditioning: mode=sampler_delta_conditioning experimental regions={len(stack.regions)}")
    for index, regional in enumerate(stack.regions, start=1):
        report.add(_region_report_line(index, regional, "sampler_delta_conditioning"))

    def cfg_function(args):
        base_noise = _base_cfg_noise(args, previous)
        x = args["input"]
        sigma = args["sigma"]
        comfy_model = args["model"]
        model_options = _regional_model_options(args.get("model_options", {}))
        global_pred = args["cond_denoised"]
        regional_noise = base_noise
        for regional in stack.regions:
            conditioning = _combine_conditioning(stack.global_conditioning, regional.conditioning)
            region_pred = _calc_regional_pred(comfy_model, conditioning, x, sigma, model_options)
            delta = region_pred - global_pred
            mask = _mask_for_region(regional, x)
            regional_noise = regional_noise - delta.to(dtype=regional_noise.dtype) * mask * float(regional.strength)
        return regional_noise

    model_out.set_model_sampler_cfg_function(cfg_function)
    return model_out, report.text()


def _make_reset_wrapper(state: RegionalPromptRuntimeState):
    def wrapper(executor, *args, **kwargs):
        state.reset()
        return executor(*args, **kwargs)

    return wrapper


def _install_prompt_attention_override(model_out: Any, state: RegionalPromptRuntimeState) -> None:
    model_options = getattr(model_out, "model_options", None)
    if not isinstance(model_options, dict):
        return
    transformer_options = model_options.setdefault("transformer_options", {})
    previous = transformer_options.get("optimized_attention_override")

    def override(func, *args, **kwargs):
        target = func if previous is None else (lambda *a, **kw: previous(func, *a, **kw))
        return state.attention_override(target, *args, **kwargs)

    transformer_options["optimized_attention_override"] = override


def conditioning_debug_preview(stack: KreaRegionalConditioningStack) -> tuple[torch.Tensor, str]:
    if not stack.regions:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32), "No regional conditioning entries."
    width, height = stack.regions[0].region.image_size
    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    colors = [
        (1.0, 0.1, 0.1),
        (0.1, 0.8, 1.0),
        (0.2, 1.0, 0.2),
        (1.0, 0.8, 0.1),
        (0.8, 0.2, 1.0),
        (1.0, 0.4, 0.0),
    ]
    lines = [f"regional_conditioning_regions={len(stack.regions)}"]
    for index, regional in enumerate(stack.regions, start=1):
        mask = _pixel_mask_for_region(regional)[:1].clamp(0.0, 1.0)
        color = torch.tensor(colors[(index - 1) % len(colors)], dtype=image.dtype).view(1, 1, 1, 3)
        image = torch.maximum(image, mask.unsqueeze(-1) * color)
        lines.append(_region_report_line(index, regional, "debug"))
    return image.clamp(0.0, 1.0), "\n".join(lines)


def region_ids_match_lora(conditioning_stack: KreaRegionalConditioningStack, lora_stack: Any) -> str:
    cond_ids = [r.region.region_id for r in conditioning_stack.regions]
    lora_ids = [r.region.region_id for r in getattr(lora_stack, "regions", ())]
    lines = [f"conditioning_region_ids={cond_ids}", f"lora_region_ids={lora_ids}"]
    for prompt in conditioning_stack.regions:
        matching = [r for r in getattr(lora_stack, "regions", ()) if r.region.region_id == prompt.region.region_id]
        prompt_stats = mask_stats(prompt.region.token_mask.squeeze(-1), offset=0)
        if matching:
            lora = matching[0]
            lora_stats = mask_stats(lora.region.token_mask.squeeze(-1), offset=0)
            aligns = torch.equal(prompt.region.token_mask, lora.region.token_mask)
            lines.append(
                f"region_id={prompt.region.region_id} shared_with_lora=True prompt_region_id={prompt.region.region_id} "
                f"lora_region_id={lora.region.region_id} image_token_mask_shape={prompt_stats.shape} "
                f"prompt_mask_range={prompt_stats.nonzero_range} lora_mask_range={lora_stats.nonzero_range} "
                f"mask_min={prompt_stats.min:.4f} mask_max={prompt_stats.max:.4f} mask_mean={prompt_stats.mean:.4f} "
                f"mask_checksum={prompt_stats.checksum:.4f} prompt_lora_mask_align={aligns}"
            )
        else:
            lines.append(f"region_id={prompt.region.region_id} shared_with_lora=False")
    return "\n".join(lines)


def _combine_conditioning(global_conditioning: Any, regional_conditioning: Any) -> Any:
    return list(global_conditioning) + list(regional_conditioning)


def _attach_region_metadata(
    conditioning: Any,
    region: Any,
    strength: float,
    outside_strength: float,
    feather_px: int | None,
    prompt_attention_mode: PromptAttentionMode,
    prompt_attention_strength: float,
) -> Any:
    out = []
    for item in conditioning:
        cond = item[0]
        meta = item[1].copy() if len(item) > 1 and isinstance(item[1], dict) else {}
        meta["krea_region"] = {
            "region_id": region.region_id,
            "pixel_bbox": region.pixel_bbox,
            "normalized_bbox": region.normalized_bbox,
            "image_size": region.image_size,
            "feather_px": region.feather_px if feather_px is None else feather_px,
            "strength": strength,
            "outside_strength": outside_strength,
            "prompt_attention_mode": prompt_attention_mode,
            "prompt_attention_strength": prompt_attention_strength,
        }
        out.append([cond, meta])
    return out


def _base_cfg_noise(args: dict[str, Any], previous: Any) -> torch.Tensor:
    if previous is not None:
        return previous(args)
    return args["uncond"] + float(args["cond_scale"]) * (args["cond"] - args["uncond"])


def _calc_regional_pred(model: Any, conditioning: Any, x: torch.Tensor, sigma: torch.Tensor, model_options: dict[str, Any]) -> torch.Tensor:
    import comfy.samplers  # type: ignore

    return comfy.samplers.calc_cond_batch(model, [conditioning], x, sigma, model_options)[0]


def _regional_model_options(model_options: dict[str, Any]) -> dict[str, Any]:
    out = dict(model_options)
    out.pop("sampler_cfg_function", None)
    return out


def _mask_for_region(regional: KreaRegionalConditioning, target: torch.Tensor) -> torch.Tensor:
    mask = _pixel_mask_for_region(regional).unsqueeze(1)
    if tuple(mask.shape[-2:]) != tuple(target.shape[-2:]):
        mask = F.interpolate(mask, size=target.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    if mask.shape[0] == 1 and target.shape[0] > 1:
        mask = mask.repeat(target.shape[0], 1, 1, 1)
    elif mask.shape[0] != target.shape[0]:
        mask = mask[:1].repeat(target.shape[0], 1, 1, 1)
    mask = mask.to(device=target.device, dtype=target.dtype)
    if regional.outside_strength:
        mask = mask + (1.0 - mask) * float(regional.outside_strength)
    return mask


def _pixel_mask_for_region(regional: KreaRegionalConditioning) -> torch.Tensor:
    if regional.feather_px is None or int(regional.feather_px) == int(regional.region.feather_px):
        return regional.region.pixel_mask
    width, height = regional.region.image_size
    return make_pixel_mask(
        regional.region.pixel_bbox,
        width=width,
        height=height,
        feather_px=int(regional.feather_px),
        batch_size=max(1, int(regional.region.pixel_mask.shape[0])),
    )


def _region_report_line(index: int, regional: KreaRegionalConditioning, prefix: str) -> str:
    mask = _pixel_mask_for_region(regional)
    latent = regional.region.latent_mask
    token = regional.region.token_mask
    return (
        f"[{index}] {prefix} region_id={regional.region.region_id} pixel_bbox={regional.region.pixel_bbox} "
        f"normalized_bbox={regional.region.normalized_bbox} latent_shape={tuple(latent.shape)} token_shape={tuple(token.shape)} "
        f"mask_min={float(mask.min()):.4f} mask_max={float(mask.max()):.4f} mask_mean={float(mask.mean()):.4f} "
        f"strength={regional.strength:.3f} outside_strength={regional.outside_strength:.3f} "
        f"prompt_attention_mode={getattr(regional, 'prompt_attention_mode', 'forbid')} "
        f"prompt_attention_strength={float(getattr(regional, 'prompt_attention_strength', 5.0)):.3f}"
    )


def _conditioning_tensor(conditioning: Any) -> torch.Tensor | None:
    for item in conditioning or []:
        if isinstance(item, (list, tuple)) and item and torch.is_tensor(item[0]):
            return item[0]
        if torch.is_tensor(item):
            return item
    return None


def _fit_context_tensor(tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise RuntimeError(f"Regional prompt conditioning tensor must be rank 3, got {tuple(tensor.shape)}")
    if int(tensor.shape[-1]) != int(context.shape[-1]):
        raise RuntimeError(f"Regional prompt conditioning feature dim {tensor.shape[-1]} does not match runtime context dim {context.shape[-1]}")
    out = tensor.to(device=context.device, dtype=context.dtype)
    if out.shape[0] == context.shape[0]:
        return out
    if out.shape[0] == 1:
        return out.repeat(context.shape[0], 1, 1)
    if context.shape[0] == 1:
        return out[:1]
    raise RuntimeError(f"Regional prompt conditioning batch {out.shape[0]} does not match runtime context batch {context.shape[0]}")


def _extract_context(args: list[Any], kwargs: dict[str, Any]) -> tuple[Any, str | int]:
    if "context" in kwargs:
        return kwargs["context"], "context"
    if len(args) > 2:
        return args[2], 2
    return None, "context"


def _replace_context(args: list[Any], kwargs: dict[str, Any], source: str | int, context: torch.Tensor) -> None:
    if source == "context":
        kwargs["context"] = context
    elif isinstance(source, int):
        args[source] = context
    else:
        kwargs["context"] = context


def _prompt_mode_strength(mode: str, strength: float, outside_strength: float) -> float:
    if mode == "none":
        return 0.0
    outside = max(0.0, min(1.0, float(outside_strength)))
    if mode == "forbid":
        return FORBID_BIAS * (1.0 - outside)
    return max(0.0, float(strength)) * (1.0 - outside)
