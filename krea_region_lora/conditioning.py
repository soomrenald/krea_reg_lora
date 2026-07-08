from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .masks import make_pixel_mask
from .types import KreaRegionalConditioning, KreaRegionalConditioningStack


@dataclass
class RegionalConditioningReport:
    lines: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines)


def encode_regional_conditioning(clip: Any, region: Any, text: str, strength: float, outside_strength: float, feather: int) -> KreaRegionalConditioning:
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None")
    tokens = clip.tokenize(text)
    conditioning = _attach_region_metadata(
        clip.encode_from_tokens_scheduled(tokens),
        region,
        float(strength),
        float(outside_strength),
        None if int(feather) < 0 else int(feather),
    )
    return KreaRegionalConditioning(
        region=region,
        conditioning=conditioning,
        text=str(text),
        strength=float(strength),
        outside_strength=float(outside_strength),
        feather_px=None if int(feather) < 0 else int(feather),
    )


def build_conditioning_stack(global_conditioning: Any, regions: list[KreaRegionalConditioning]) -> KreaRegionalConditioningStack:
    return KreaRegionalConditioningStack(global_conditioning=global_conditioning, regions=tuple(regions))


def build_regional_conditioning_model(model: Any, stack: KreaRegionalConditioningStack, *, debug: bool = False) -> tuple[Any, str]:
    if not stack.regions:
        return model, "No regional conditioning entries; model unchanged."

    model_out = model.clone()
    previous = getattr(model_out, "model_options", {}).get("sampler_cfg_function")
    report = RegionalConditioningReport()
    report.add(f"Regional prompt conditioning: regions={len(stack.regions)}")
    for index, regional in enumerate(stack.regions, start=1):
        report.add(_region_report_line(index, regional, "install"))

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
    for region_id in cond_ids:
        lines.append(f"region_id={region_id} shared_with_lora={region_id in lora_ids}")
    return "\n".join(lines)


def _combine_conditioning(global_conditioning: Any, regional_conditioning: Any) -> Any:
    return list(global_conditioning) + list(regional_conditioning)


def _attach_region_metadata(conditioning: Any, region: Any, strength: float, outside_strength: float, feather_px: int | None) -> Any:
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
        f"strength={regional.strength:.3f} outside_strength={regional.outside_strength:.3f}"
    )
