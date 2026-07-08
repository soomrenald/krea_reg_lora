from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .types import BatchMode, BBoxFormat, KreaRegion


KREA_VAE_SCALE = 8
KREA_TOKEN_PIXELS = 16


def coerce_bbox_list(bboxes: Any) -> list[tuple[float, float, float, float]]:
    """Accept KJ/Comfy BoundingBox shapes and common raw bbox variants."""
    if bboxes is None:
        return []
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().tolist()
    if isinstance(bboxes, dict):
        for keys in (
            ("x", "y", "width", "height"),
            ("x_min", "y_min", "width", "height"),
            ("x", "y", "w", "h"),
            ("left", "top", "width", "height"),
        ):
            if all(k in bboxes for k in keys):
                return [tuple(float(bboxes[k]) for k in keys)]  # type: ignore[arg-type]
        if all(k in bboxes for k in ("x0", "y0", "x1", "y1")):
            return [(float(bboxes["x0"]), float(bboxes["y0"]), float(bboxes["x1"]), float(bboxes["y1"]))]
        if "bbox" in bboxes:
            return coerce_bbox_list(bboxes["bbox"])
        if "bboxes" in bboxes:
            return coerce_bbox_list(bboxes["bboxes"])
    if _is_bbox_tuple(bboxes):
        return [tuple(float(v) for v in bboxes[:4])]  # type: ignore[index]
    if isinstance(bboxes, Sequence) and not isinstance(bboxes, (str, bytes)):
        out: list[tuple[float, float, float, float]] = []
        for item in bboxes:
            out.extend(coerce_bbox_list(item))
        return out
    return []


def infer_image_size(latent: dict[str, Any] | None, width: int | None, height: int | None) -> tuple[int, int, int]:
    if latent is not None and isinstance(latent.get("samples"), torch.Tensor):
        samples = latent["samples"]
        if samples.ndim >= 4:
            return int(samples.shape[-1] * KREA_VAE_SCALE), int(samples.shape[-2] * KREA_VAE_SCALE), int(samples.shape[0])
    if width is None or height is None:
        raise ValueError("Either latent or width/height must be provided")
    return int(width), int(height), 1


def region_from_bbox(
    bboxes: Any,
    *,
    width: int,
    height: int,
    bbox_index: int = 0,
    bbox_format: BBoxFormat = "xywh",
    grow_px: int = 0,
    feather_px: int = 32,
    snap_to_krea_token_grid: bool = True,
    batch_mode: BatchMode = "repeat",
    batch_size: int = 1,
) -> KreaRegion:
    bbox_list = coerce_bbox_list(bboxes)
    if bbox_list:
        index = _resolve_bbox_index(int(bbox_index), len(bbox_list))
        pixel_bbox = normalize_bbox(
            bbox_list[index],
            width=width,
            height=height,
            bbox_format=bbox_format,
            grow_px=grow_px,
            snap_to_krea_token_grid=snap_to_krea_token_grid,
        )
    else:
        pixel_bbox = (0, 0, 0, 0)

    mask_batch = int(batch_size) if batch_mode == "per_batch" else 1
    pixel_mask = make_pixel_mask(pixel_bbox, width=width, height=height, feather_px=feather_px, batch_size=mask_batch)
    latent_mask = pixel_to_latent_mask(pixel_mask)
    token_mask = pixel_to_token_mask(pixel_mask)
    return KreaRegion(
        region_id=f"bbox_{int(bbox_index)}_{pixel_bbox[0]}_{pixel_bbox[1]}_{pixel_bbox[2]}_{pixel_bbox[3]}",
        pixel_bbox=pixel_bbox,
        normalized_bbox=_normalized_bbox(pixel_bbox, int(width), int(height)),
        image_size=(int(width), int(height)),
        feather_px=int(feather_px),
        pixel_mask=pixel_mask,
        latent_mask=latent_mask,
        token_mask=token_mask,
        bbox_index=int(bbox_index),
        bbox_format=bbox_format,
        batch_mode=batch_mode,
        metadata={
            "grow_px": int(grow_px),
            "feather_px": int(feather_px),
            "snap_to_krea_token_grid": bool(snap_to_krea_token_grid),
            "bbox_count": len(bbox_list),
        },
    )


def normalize_bbox(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    bbox_format: BBoxFormat,
    grow_px: int,
    snap_to_krea_token_grid: bool,
) -> tuple[int, int, int, int]:
    x0, y0, a, b = bbox
    if all(0.0 <= value <= 1.0 for value in (x0, y0, a, b)):
        x0 *= width
        a *= width
        y0 *= height
        b *= height
    if bbox_format == "xywh":
        x1 = x0 + max(0.0, a)
        y1 = y0 + max(0.0, b)
    elif bbox_format == "xyxy":
        x1 = a
        y1 = b
    else:
        raise ValueError(f"Unsupported bbox_format {bbox_format}")

    grow = int(grow_px)
    x0, y0, x1, y1 = x0 - grow, y0 - grow, x1 + grow, y1 + grow
    if snap_to_krea_token_grid:
        grid = KREA_TOKEN_PIXELS
        x0 = math.floor(x0 / grid) * grid
        y0 = math.floor(y0 / grid) * grid
        x1 = math.ceil(x1 / grid) * grid
        y1 = math.ceil(y1 / grid) * grid
    ix0 = max(0, min(int(math.floor(x0)), int(width)))
    iy0 = max(0, min(int(math.floor(y0)), int(height)))
    ix1 = max(0, min(int(math.ceil(x1)), int(width)))
    iy1 = max(0, min(int(math.ceil(y1)), int(height)))
    if ix1 <= ix0 or iy1 <= iy0:
        return (0, 0, 0, 0)
    return (ix0, iy0, ix1, iy1)


def _normalized_bbox(pixel_bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = pixel_bbox
    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(x0) / float(width), float(y0) / float(height), float(x1) / float(width), float(y1) / float(height))


def make_pixel_mask(pixel_bbox: tuple[int, int, int, int], *, width: int, height: int, feather_px: int, batch_size: int) -> torch.Tensor:
    x0, y0, x1, y1 = pixel_bbox
    mask = torch.zeros((1, int(height), int(width)), dtype=torch.float32)
    if x1 > x0 and y1 > y0:
        mask[:, y0:y1, x0:x1] = 1.0
        feather = max(0, int(feather_px))
        if feather > 0:
            mask = _feather_inside(mask, feather)
    return mask.repeat(max(1, int(batch_size)), 1, 1)


def pixel_to_latent_mask(pixel_mask: torch.Tensor) -> torch.Tensor:
    h = max(1, pixel_mask.shape[-2] // KREA_VAE_SCALE)
    w = max(1, pixel_mask.shape[-1] // KREA_VAE_SCALE)
    return F.interpolate(pixel_mask.unsqueeze(1), size=(h, w), mode="area").clamp(0.0, 1.0)


def pixel_to_token_mask(pixel_mask: torch.Tensor) -> torch.Tensor:
    h = max(1, pixel_mask.shape[-2] // KREA_TOKEN_PIXELS)
    w = max(1, pixel_mask.shape[-1] // KREA_TOKEN_PIXELS)
    token_grid = F.interpolate(pixel_mask.unsqueeze(1), size=(h, w), mode="area").clamp(0.0, 1.0)
    return token_grid.flatten(2).transpose(1, 2)


def debug_bbox_image(region: KreaRegion) -> torch.Tensor:
    width, height = region.image_size
    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    mask = region.pixel_mask[:1]
    image[..., 1] = mask
    x0, y0, x1, y1 = region.pixel_bbox
    if x1 > x0 and y1 > y0:
        image[:, y0:y1, x0:min(x0 + 2, x1), 0] = 1.0
        image[:, y0:y1, max(x0, x1 - 2):x1, 0] = 1.0
        image[:, y0:min(y0 + 2, y1), x0:x1, 0] = 1.0
        image[:, max(y0, y1 - 2):y1, x0:x1, 0] = 1.0
    return image.clamp(0.0, 1.0)


def _resolve_bbox_index(bbox_index: int, bbox_count: int) -> int:
    if bbox_count <= 0 or bbox_index <= 0:
        return 0
    return max(0, min(bbox_index - 1, bbox_count - 1))


def _feather_inside(mask: torch.Tensor, feather_px: int) -> torch.Tensor:
    kernel = 2 * feather_px + 1
    pooled = F.avg_pool2d(mask.unsqueeze(1), kernel_size=kernel, stride=1, padding=feather_px, count_include_pad=False).squeeze(1)
    return torch.minimum(mask, pooled * (kernel * kernel) / max(1, (feather_px + 1) ** 2)).clamp(0.0, 1.0)


def _is_bbox_tuple(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4 and all(
        isinstance(v, (int, float)) for v in value[:4]
    )
