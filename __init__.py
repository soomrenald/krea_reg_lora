from .nodes import (
    KreaBBoxToRegionalTokens,
    KreaRegionalConditioningApply,
    KreaRegionalConditioningDebug,
    KreaRegionalConditioningStack,
    KreaRegionalPrompt,
    KreaRegionalPromptLoRAMatchDebug,
    KreaRegionalLoRA,
    KreaRegionalLoRAApply,
    KreaRegionalLoRADebug,
    KreaRegionalLoRAStack,
)


NODE_CLASS_MAPPINGS = {
    "KreaBBoxToRegionalTokens": KreaBBoxToRegionalTokens,
    "KreaRegionalPrompt": KreaRegionalPrompt,
    "KreaRegionalConditioningStack": KreaRegionalConditioningStack,
    "KreaRegionalConditioningApply": KreaRegionalConditioningApply,
    "KreaRegionalConditioningDebug": KreaRegionalConditioningDebug,
    "KreaRegionalPromptLoRAMatchDebug": KreaRegionalPromptLoRAMatchDebug,
    "KreaRegionalLoRA": KreaRegionalLoRA,
    "KreaRegionalLoRAStack": KreaRegionalLoRAStack,
    "KreaRegionalLoRAApply": KreaRegionalLoRAApply,
    "KreaRegionalLoRADebug": KreaRegionalLoRADebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaBBoxToRegionalTokens": "Krea BBox To Regional Tokens",
    "KreaRegionalPrompt": "Krea Regional Prompt",
    "KreaRegionalConditioningStack": "Krea Regional Conditioning Stack",
    "KreaRegionalConditioningApply": "Krea Regional Conditioning Apply",
    "KreaRegionalConditioningDebug": "Krea Regional Conditioning Debug",
    "KreaRegionalPromptLoRAMatchDebug": "Krea Regional Prompt LoRA Match Debug",
    "KreaRegionalLoRA": "Krea Regional LoRA",
    "KreaRegionalLoRAStack": "Krea Regional LoRA Stack",
    "KreaRegionalLoRAApply": "Krea Regional LoRA Apply",
    "KreaRegionalLoRADebug": "Krea Regional LoRA Debug",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
