from .nodes import (
    K2BBoxToRegionalMask,
    K2RegionalAttentionLoRASampler,
    K2RegionalCharacterLoRA,
    K2RegionalDecodeComposite,
    K2RegionalLayerLoRAApply,
    K2RegionalLoRAStack3,
)

NODE_CLASS_MAPPINGS = {
    "K2BBoxToRegionalMask": K2BBoxToRegionalMask,
    "K2RegionalCharacterLoRA": K2RegionalCharacterLoRA,
    "K2RegionalLoRAStack3": K2RegionalLoRAStack3,
    "K2RegionalLayerLoRAApply": K2RegionalLayerLoRAApply,
    "K2RegionalAttentionLoRASampler": K2RegionalAttentionLoRASampler,
    "K2RegionalDecodeComposite": K2RegionalDecodeComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K2BBoxToRegionalMask": "K2 BBox To Regional Mask",
    "K2RegionalCharacterLoRA": "K2 Regional Character LoRA",
    "K2RegionalLoRAStack3": "K2 Regional LoRA Stack 3",
    "K2RegionalLayerLoRAApply": "K2 Regional Layer LoRA Apply",
    "K2RegionalAttentionLoRASampler": "K2 Regional Attention LoRA Sampler",
    "K2RegionalDecodeComposite": "K2 Regional Decode Composite",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
