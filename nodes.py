from __future__ import annotations

from typing import Any

try:
    from .krea_region_lora.layer_injection import build_layer_injection_model
    from .krea_region_lora.masks import debug_bbox_image, infer_image_size, region_from_bbox
    from .krea_region_lora.types import MEASUREMENT_SOURCE_OPTIONS, KreaRegionalLora, KreaRegionalLoraStack, parse_measurement_sources
except ImportError:
    from krea_region_lora.layer_injection import build_layer_injection_model
    from krea_region_lora.masks import debug_bbox_image, infer_image_size, region_from_bbox
    from krea_region_lora.types import MEASUREMENT_SOURCE_OPTIONS, KreaRegionalLora, KreaRegionalLoraStack, parse_measurement_sources

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - local tests run outside ComfyUI
    folder_paths = None  # type: ignore


def _lora_names() -> list[str]:
    if folder_paths is None:
        return ["None"]
    names = folder_paths.get_filename_list("loras")
    return ["None"] + [n for n in names if n != "None"]


class KreaBBoxToRegionalTokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "bbox_index": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "bbox_format": (["xywh", "xyxy"], {"default": "xywh"}),
                "grow_px": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "feather_px": ("INT", {"default": 32, "min": 0, "max": 2048}),
                "snap_to_krea_token_grid": ("BOOLEAN", {"default": True}),
                "batch_mode": (["repeat", "single", "per_batch"], {"default": "repeat"}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX",),
                "kj_bboxes": ("BBOX",),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("MASK", "KREA_REGION", "IMAGE", "STRING")
    RETURN_NAMES = ("region_mask", "region", "debug_bbox_image", "debug")
    FUNCTION = "build"
    CATEGORY = "Krea/Regional LoRA"

    def build(
        self,
        width,
        height,
        bbox_index=0,
        bbox_format="xywh",
        grow_px=0,
        feather_px=32,
        snap_to_krea_token_grid=True,
        batch_mode="repeat",
        bboxes=None,
        kj_bboxes=None,
        latent=None,
    ):
        bbox_source = bboxes if bboxes is not None else kj_bboxes
        image_w, image_h, batch = infer_image_size(latent, width, height)
        region = region_from_bbox(
            bbox_source,
            width=image_w,
            height=image_h,
            bbox_index=bbox_index,
            bbox_format=bbox_format,
            grow_px=grow_px,
            feather_px=feather_px,
            snap_to_krea_token_grid=snap_to_krea_token_grid,
            batch_mode=batch_mode,
            batch_size=batch,
        )
        debug = (
            f"bbox={region.pixel_bbox} image_size={region.image_size} "
            f"bbox_count={region.metadata.get('bbox_count', 0)} token_count={region.token_mask.shape[1]}"
        )
        return (region.pixel_mask, region, debug_bbox_image(region), debug)


class KreaRegionalLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region": ("KREA_REGION",),
                "lora_name": (_lora_names(),),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "delta_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
                "ignore_text_encoder_lora": ("BOOLEAN", {"default": True}),
                "measurement_sources": (MEASUREMENT_SOURCE_OPTIONS, {"default": "direct_delta"}),
                "normalization": (["relative_norm", "percentile", "minmax", "raw"], {"default": "relative_norm"}),
                "percentile": ("FLOAT", {"default": 95.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "modified_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 100.0, "step": 0.001}),
                "retention": (["sticky", "decay", "instant"], {"default": "sticky"}),
                "decay": ("FLOAT", {"default": 0.96, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("KREA_REGIONAL_LORA",)
    RETURN_NAMES = ("regional_lora",)
    FUNCTION = "bind"
    CATEGORY = "Krea/Regional LoRA"

    def bind(
        self,
        region,
        lora_name,
        lora_strength=1.0,
        delta_strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
        enabled=True,
        ignore_text_encoder_lora=True,
        measurement_sources="direct_delta",
        normalization="relative_norm",
        percentile=95.0,
        modified_threshold=0.05,
        retention="sticky",
        decay=0.96,
    ):
        return (
            KreaRegionalLora(
                region=region,
                lora_name=lora_name,
                lora_strength=float(lora_strength),
                delta_strength=float(delta_strength),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                enabled=bool(enabled),
                ignore_text_encoder_lora=bool(ignore_text_encoder_lora),
                measurement_sources=parse_measurement_sources(measurement_sources),
                normalization=normalization,
                percentile=float(percentile),
                threshold=float(modified_threshold),
                retention=retention,
                decay=float(decay),
            ),
        )


class KreaRegionalLoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regional_lora_1": ("KREA_REGIONAL_LORA",),
                "overlap_mode": (["normalize", "priority_1", "priority_last", "add"], {"default": "normalize"}),
                "attention_isolation_strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "cross_lora_mode": (["allow", "penalize", "block"], {"default": "penalize"}),
                "cross_lora_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
            },
            "optional": {
                "regional_lora_2": ("KREA_REGIONAL_LORA",),
                "regional_lora_3": ("KREA_REGIONAL_LORA",),
                "regional_lora_4": ("KREA_REGIONAL_LORA",),
                "regional_lora_5": ("KREA_REGIONAL_LORA",),
                "regional_lora_6": ("KREA_REGIONAL_LORA",),
            },
        }

    RETURN_TYPES = ("KREA_REGIONAL_LORA_STACK",)
    RETURN_NAMES = ("regional_lora_stack",)
    FUNCTION = "stack"
    CATEGORY = "Krea/Regional LoRA"

    def stack(self, regional_lora_1, overlap_mode="normalize", attention_isolation_strength=5.0, cross_lora_mode="penalize", cross_lora_strength=3.0, **kwargs):
        regions = [regional_lora_1]
        for key in ("regional_lora_2", "regional_lora_3", "regional_lora_4", "regional_lora_5", "regional_lora_6"):
            value = kwargs.get(key)
            if value is not None:
                regions.append(value)
        return (
            KreaRegionalLoraStack(
                regions=tuple(regions),
                overlap_mode=overlap_mode,
                attention_isolation_strength=float(attention_isolation_strength),
                cross_lora_mode=cross_lora_mode,
                cross_lora_strength=float(cross_lora_strength),
            ),
        )


class KreaRegionalLoRAApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regional_lora_stack": ("KREA_REGIONAL_LORA_STACK",),
                "layer_targets": (["attn_out_mlp", "attention_only", "all_matched_linears"], {"default": "attn_out_mlp"}),
                "outside_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_token_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "Krea/Regional LoRA"

    def apply(self, model, regional_lora_stack, layer_targets="attn_out_mlp", outside_strength=0.0, text_token_strength=0.0, debug_logging=False):
        return build_layer_injection_model(
            model,
            regional_lora_stack,
            target_policy=layer_targets,
            outside_strength=float(outside_strength),
            text_token_strength=float(text_token_strength),
            debug=bool(debug_logging),
        )


class KreaRegionalLoRADebug:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"regional_lora_stack": ("KREA_REGIONAL_LORA_STACK",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "debug"
    CATEGORY = "Krea/Regional LoRA"

    def debug(self, regional_lora_stack):
        lines = [
            f"regions={len(regional_lora_stack.regions)} enabled={len(regional_lora_stack.enabled_regions)}",
            f"attention_isolation_strength={regional_lora_stack.attention_isolation_strength}",
            f"cross_lora_mode={regional_lora_stack.cross_lora_mode}",
        ]
        for i, regional in enumerate(regional_lora_stack.regions, start=1):
            lines.append(
                f"[{i}] lora={regional.lora_name} bbox={regional.region.pixel_bbox} "
                f"sources={','.join(regional.measurement_sources)} threshold={regional.threshold}"
            )
        return ("\n".join(lines),)
