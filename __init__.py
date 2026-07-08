from .nodes import (
    KreaBBoxToRegionalTokens,
    KreaRegionalLoRA,
    KreaRegionalLoRAApply,
    KreaRegionalLoRADebug,
    KreaRegionalLoRAStack,
)


NODE_CLASS_MAPPINGS = {
    "KreaBBoxToRegionalTokens": KreaBBoxToRegionalTokens,
    "KreaRegionalLoRA": KreaRegionalLoRA,
    "KreaRegionalLoRAStack": KreaRegionalLoRAStack,
    "KreaRegionalLoRAApply": KreaRegionalLoRAApply,
    "KreaRegionalLoRADebug": KreaRegionalLoRADebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaBBoxToRegionalTokens": "Krea BBox To Regional Tokens",
    "KreaRegionalLoRA": "Krea Regional LoRA",
    "KreaRegionalLoRAStack": "Krea Regional LoRA Stack",
    "KreaRegionalLoRAApply": "Krea Regional LoRA Apply",
    "KreaRegionalLoRADebug": "Krea Regional LoRA Debug",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
