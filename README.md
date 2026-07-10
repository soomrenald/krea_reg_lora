# Krea Regional LoRA Attention

ComfyUI custom nodes for regional LoRA application on Krea-style transformer diffusion models.

The main node applies LoRA deltas only to selected image-token regions, tracks which image tokens were actually modified, and injects an asymmetric attention bias so unmodified image tokens are penalized or blocked from attending to LoRA-modified image tokens.

## Nodes

- `Krea BBox To Regional Tokens`
- `Krea Regional Prompt`
- `Krea Regional Conditioning Stack`
- `Krea Regional Conditioning Apply`
- `Krea Regional Conditioning Debug`
- `Krea Regional Prompt LoRA Match Debug`
- `Krea Regional LoRA`
- `Krea Regional LoRA Stack`
- `Krea Regional LoRA Apply`
- `Krea Regional LoRA Debug`

## KJNodes Ideogram 4 Compatibility

KJNodes `Ideogram 4 Prompt Builder KJ` outputs `bboxes` as Comfy `BoundingBox` data:

```python
[[{"x": 128, "y": 96, "width": 384, "height": 640}, ...]]
```

Connect that `bboxes` output to `Krea BBox To Regional Tokens` input `bboxes`.

`bbox_index` follows the visual numbering used by the KJ editor: `1` is the first box, `2` is the second box, and so on. `0` is accepted as an alias for the first box.

## Workflow

1. Draw regions in `Ideogram 4 Prompt Builder KJ`.
2. Connect its `bboxes`, `width`, and `height` to one `Krea BBox To Regional Tokens` node per region.
3. Use the global prompt only for scene, composition, lighting, and background. Do not put region subjects in the global prompt.
4. Feed each `KREA_REGION` to both matching nodes: `Krea Regional Prompt` and `Krea Regional LoRA`.
5. Combine regional prompts with `Krea Regional Conditioning Stack`, using the normal global `CLIPTextEncode` output as `global_conditioning`.
6. Combine regional LoRAs with `Krea Regional LoRA Stack`.
7. Patch the model with `Krea Regional Conditioning Apply`, then `Krea Regional LoRA Apply`, before the KSampler.
8. Connect `Krea Regional Conditioning Debug` to preview the actual soft region masks, and `Krea Regional Prompt LoRA Match Debug` to verify that prompt and LoRA entries share the same `region_id`.

Example prompt split:

- Global prompt: `cinematic two-person portrait, indoor studio, balanced composition, soft background, detailed lighting`
- Region 1 prompt: `woman, left face, detailed eyes, natural skin texture, matching pose`
- Region 2 prompt: `man, smaller right face, detailed eyes, natural skin texture, matching pose`

## Important Controls

- `measurement_sources`: delta source selector. `direct_delta` is the default; `hidden_state_delta` measures the regional layer output against its input hidden state.
- `normalization`: `relative_norm`, `percentile`, `minmax`, or `raw`.
- `modified_threshold`: token score threshold for marking a token modified by that LoRA.
- `retention`: `sticky`, `decay`, or `instant`.
- `conditioning_mode`: `transformer_attention_bias` is the default. `sampler_delta_conditioning` is an experimental opt-in fallback.
- `prompt_attention_mode`: `none`, `penalize`, or `forbid` regional prompt-token leakage outside the matching image-token mask.
- `prompt_attention_strength`: penalty strength used by `prompt_attention_mode=penalize`.
- `attention_isolation_mode`: `none`, `penalize`, or `forbid` unmodified image-token queries attending to LoRA-modified image-token keys.
- `attention_isolation_strength`: penalty strength used by `attention_isolation_mode=penalize`.
- `modified_outward_mode`: separately controls LoRA-modified image-token queries attending outward.
- `cross_lora_mode`: `allow`, `penalize`, or `block` attention between different LoRA-modified token sets.

## Implementation Notes

The package uses ComfyUI model patcher wrappers and `transformer_options["optimized_attention_override"]`, which are present in current ComfyUI. It does not require running a custom sampler.

Regional prompts default to transformer attention bias. Each regional prompt conditioning tensor is appended as real Krea2 text tokens in the runtime self-attention sequence, its prompt-token range is tracked, and image-token queries outside the matching `KREA_REGION` mask are penalized or forbidden from attending to that regional prompt-token range. `sampler_delta_conditioning` remains available only as an explicitly selected experimental fallback.

The model patch is conservative:

- Text-token LoRA application defaults to `0.0`.
- Outside-region LoRA delta application defaults to `0.0`.
- Text encoder LoRA keys are ignored by default.
- Regional prompt attention bias and LoRA modified-token isolation are separate additive attention-bias masks.
- Attention masking only activates when the attention call is self-attention over a sequence length that matches the tracked text/image/padding layout.

## Local Checks

These tests do not launch ComfyUI:

```bash
python run_tests.py
```
