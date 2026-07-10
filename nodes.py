from __future__ import annotations

from typing import Any

try:
    from .krea_region_lora.conditioning import build_conditioning_stack, build_regional_conditioning_model, conditioning_debug_preview, encode_regional_conditioning, region_ids_match_lora
    from .krea_region_lora.layer_injection import build_layer_injection_model
    from .krea_region_lora.masks import debug_bbox_image, infer_image_size, region_from_bbox
    from .krea_region_lora.types import MEASUREMENT_SOURCE_OPTIONS, KreaRegionalLora, KreaRegionalLoraStack, parse_measurement_sources
except ImportError:
    from krea_region_lora.conditioning import build_conditioning_stack, build_regional_conditioning_model, conditioning_debug_preview, encode_regional_conditioning, region_ids_match_lora
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
            f"region_id={region.region_id} bbox={region.pixel_bbox} normalized_bbox={region.normalized_bbox} "
            f"image_size={region.image_size} feather_px={region.feather_px} "
            f"bbox_count={region.metadata.get('bbox_count', 0)} token_count={region.token_mask.shape[1]} "
            f"img_len={region.img_len} text_len={region.text_len} "
            f"mask_min={float(region.pixel_mask.min()):.4f} mask_max={float(region.pixel_mask.max()):.4f} "
            f"mask_mean={float(region.pixel_mask.mean()):.4f}"
        )
        return (region.pixel_mask, region, debug_bbox_image(region), debug)


class KreaRegionalPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region": ("KREA_REGION",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "outside_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 1}),
                "prompt_attention_mode": (["none", "penalize", "forbid"], {"default": "forbid"}),
                "prompt_attention_strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("KREA_REGIONAL_CONDITIONING",)
    RETURN_NAMES = ("regional_conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Krea/Regional LoRA"

    def encode(self, region, clip, text, strength=1.0, outside_strength=0.0, feather=-1, prompt_attention_mode="forbid", prompt_attention_strength=5.0):
        return (
            encode_regional_conditioning(
                clip,
                region,
                text,
                float(strength),
                float(outside_strength),
                int(feather),
                prompt_attention_mode,
                float(prompt_attention_strength),
            ),
        )


class KreaRegionalConditioningStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_conditioning": ("CONDITIONING",),
            },
            "optional": {
                "regional_conditioning_1": ("KREA_REGIONAL_CONDITIONING",),
                "regional_conditioning_2": ("KREA_REGIONAL_CONDITIONING",),
                "regional_conditioning_3": ("KREA_REGIONAL_CONDITIONING",),
                "regional_conditioning_4": ("KREA_REGIONAL_CONDITIONING",),
                "regional_conditioning_5": ("KREA_REGIONAL_CONDITIONING",),
                "regional_conditioning_6": ("KREA_REGIONAL_CONDITIONING",),
            },
        }

    RETURN_TYPES = ("KREA_REGIONAL_CONDITIONING_STACK",)
    RETURN_NAMES = ("regional_conditioning_stack",)
    FUNCTION = "stack"
    CATEGORY = "Krea/Regional LoRA"

    def stack(self, global_conditioning, **kwargs):
        regions = []
        for key in (
            "regional_conditioning_1",
            "regional_conditioning_2",
            "regional_conditioning_3",
            "regional_conditioning_4",
            "regional_conditioning_5",
            "regional_conditioning_6",
        ):
            value = kwargs.get(key)
            if value is not None:
                regions.append(value)
        return (build_conditioning_stack(global_conditioning, regions),)


class KreaRegionalConditioningApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regional_conditioning_stack": ("KREA_REGIONAL_CONDITIONING_STACK",),
                "conditioning_mode": (["transformer_attention_bias", "sampler_delta_conditioning", "disabled"], {"default": "transformer_attention_bias"}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "Krea/Regional LoRA"

    def apply(self, model, regional_conditioning_stack, conditioning_mode="transformer_attention_bias", debug_logging=False):
        return build_regional_conditioning_model(
            model,
            regional_conditioning_stack,
            conditioning_mode=conditioning_mode,
            debug=bool(debug_logging),
        )


class KreaRegionalConditioningDebug:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"regional_conditioning_stack": ("KREA_REGIONAL_CONDITIONING_STACK",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("mask_preview", "report")
    FUNCTION = "debug"
    CATEGORY = "Krea/Regional LoRA"

    def debug(self, regional_conditioning_stack):
        return conditioning_debug_preview(regional_conditioning_stack)


class KreaRegionalPromptLoRAMatchDebug:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regional_conditioning_stack": ("KREA_REGIONAL_CONDITIONING_STACK",),
                "regional_lora_stack": ("KREA_REGIONAL_LORA_STACK",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "debug"
    CATEGORY = "Krea/Regional LoRA"

    def debug(self, regional_conditioning_stack, regional_lora_stack):
        return (region_ids_match_lora(regional_conditioning_stack, regional_lora_stack),)


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
                "attention_isolation_mode": (["none", "penalize", "forbid"], {"default": "penalize"}),
                "attention_isolation_strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "modified_outward_mode": (["none", "penalize", "forbid"], {"default": "none"}),
                "modified_outward_strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
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

    def stack(
        self,
        regional_lora_1,
        overlap_mode="normalize",
        attention_isolation_mode="penalize",
        attention_isolation_strength=5.0,
        modified_outward_mode="none",
        modified_outward_strength=5.0,
        cross_lora_mode="penalize",
        cross_lora_strength=3.0,
        **kwargs,
    ):
        regions = [regional_lora_1]
        for key in ("regional_lora_2", "regional_lora_3", "regional_lora_4", "regional_lora_5", "regional_lora_6"):
            value = kwargs.get(key)
            if value is not None:
                regions.append(value)
        return (
            KreaRegionalLoraStack(
                regions=tuple(regions),
                overlap_mode=overlap_mode,
                attention_isolation_mode=attention_isolation_mode,
                attention_isolation_strength=float(attention_isolation_strength),
                modified_outward_mode=modified_outward_mode,
                modified_outward_strength=float(modified_outward_strength),
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
            f"attention_isolation_mode={regional_lora_stack.attention_isolation_mode} attention_isolation_strength={regional_lora_stack.attention_isolation_strength}",
            f"modified_outward_mode={regional_lora_stack.modified_outward_mode} modified_outward_strength={regional_lora_stack.modified_outward_strength}",
            f"cross_lora_mode={regional_lora_stack.cross_lora_mode}",
        ]
        for i, regional in enumerate(regional_lora_stack.regions, start=1):
            lines.append(
                f"[{i}] lora={regional.lora_name} region_id={regional.region.region_id} bbox={regional.region.pixel_bbox} "
                f"sources={','.join(regional.measurement_sources)} threshold={regional.threshold}"
            )
        return ("\n".join(lines),)
