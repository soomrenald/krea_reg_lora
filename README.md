# Krea Regional LoRA Attention

ComfyUI custom nodes for regional LoRA application on Krea-style transformer diffusion models.

The main node applies LoRA deltas only to selected image-token regions, tracks which image tokens were actually modified, and injects an asymmetric attention bias so unmodified image tokens are penalized or blocked from attending to LoRA-modified image tokens.

## Nodes

- `Krea BBox To Regional Tokens`
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
2. Connect its `bboxes`, `width`, and `height` to `Krea BBox To Regional Tokens`.
3. Create one `Krea Regional LoRA` per region.
4. Combine them with `Krea Regional LoRA Stack`.
5. Run `Krea Regional LoRA Apply` on the model before your normal KSampler.

## Important Controls

- `measurement_sources`: delta source selector. `direct_delta` is the default; `hidden_state_delta` measures the regional layer output against its input hidden state.
- `normalization`: `relative_norm`, `percentile`, `minmax`, or `raw`.
- `modified_threshold`: token score threshold for marking a token modified by that LoRA.
- `retention`: `sticky`, `decay`, or `instant`.
- `attention_isolation_strength`: attention penalty from unmodified image-token queries to modified image-token keys.
- `cross_lora_mode`: `allow`, `penalize`, or `block` attention between different LoRA-modified token sets.

## Implementation Notes

The package uses ComfyUI model patcher wrappers and `transformer_options["optimized_attention_override"]`, which are present in current ComfyUI. It does not require running a custom sampler.

The model patch is conservative:

- Text-token LoRA application defaults to `0.0`.
- Outside-region LoRA delta application defaults to `0.0`.
- Text encoder LoRA keys are ignored by default.
- Attention masking only activates when the attention call is self-attention over a sequence length that matches the tracked image-token layout.

## Local Checks

These tests do not launch ComfyUI:

```bash
python run_tests.py
```
