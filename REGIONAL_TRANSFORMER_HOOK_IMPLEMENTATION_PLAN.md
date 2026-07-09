# Regional Transformer Hook Implementation Plan

## Objective

Replace the current default regional prompt implementation that patches sampler CFG output with transformer-layer/model hooks. The CFG path will remain only as an explicitly selected fallback mode named `sampler_delta_conditioning`, disabled by default.

The implementation will keep `KREA_REGION` as the shared object used by both regional LoRA and regional prompt conditioning. A single region object will own bbox geometry, image dimensions, feathering, pixel/latent/token masks, `region_id`, and token-layout metadata.

## Confirmed Krea2 Token Layout

Local ComfyUI Krea2 code uses this order in `comfy/ldm/krea2/model.py`:

```python
txtlen, imglen = context.shape[1], img.shape[1]
combined = torch.cat((context, img), dim=1)
out = final[:, txtlen:txtlen + imglen, :]
```

The implementation will therefore use:

```python
image_start = text_len
image_end = text_len + image_len
pad_len = seq_len - image_end
```

Text tokens are before image tokens. Padding, when present, is after image tokens. Region masks will never be inserted at `seq_len - image_len` unless the sequence is image-only and has no text/padding.

## Public Node/API Changes

### `KreaRegionalConditioningApply`

Add a required/defaulted mode selector:

- `conditioning_mode`: `transformer_layer_conditioning` default
- `sampler_delta_conditioning`: old CFG/sampler fallback, explicitly experimental
- `disabled`: passes model through while preserving debug stack outputs

The existing sampler-CFG code path will be renamed in code and report text to `sampler_delta_conditioning` so it is not confused with the default implementation.

### `KreaRegionalLoRAStack`

Replace the ambiguous isolation controls with explicit controls:

- `attention_isolation_mode`: `none`, `penalize`, `forbid`
- `attention_isolation_strength`: float, used only by `penalize`
- `modified_outward_mode`: `none`, `penalize`, `forbid`
- `modified_outward_strength`: float, used only by `penalize`
- keep `cross_lora_mode` and `cross_lora_strength` for region-to-region separation

`forbid` will map to a large negative attention bias. `none` will add no bias.

### `KreaRegionalLoRA`

Extend `measurement_sources`:

- `direct_delta`
- `hidden_state_delta`
- `final_prediction_delta` as optional diagnostic only

`final_prediction_delta` will not be a default source. It will be reported separately and only included in token modification scoring when explicitly selected.

## Runtime Layout Capture

The diffusion model wrapper will capture layout once per forward call before transformer hooks run:

- `text_len` from incoming `context.shape[1]`
- `image_len` from padded latent dimensions and Krea patch size: `ceil(h / patch) * ceil(w / patch)`
- `image_start = text_len`
- `image_end = text_len + image_len`
- `pad_len = seq_len - image_end`
- image grid rows/cols from padded latent dimensions

This layout will live in `RegionalRuntimeState`. `KREA_REGION` keeps static region geometry and default `img_len`; runtime state owns per-forward sequence layout because text length and padding can vary.

## Regional Prompt Conditioning In Transformer Hooks

Krea2 is a single-stream transformer: prompt tokens and image tokens are concatenated into the same self-attention sequence. There is no separate cross-attention module to patch. The replacement implementation will therefore apply regional prompt effects inside transformer block hooks using text-token conditioning deltas.

Implementation steps:

1. `KreaRegionalPrompt` continues to encode each regional prompt with CLIP.
2. `KreaRegionalConditioningApply` stores the regional conditioning stack in the model patch wrapper.
3. In the diffusion-model wrapper, prepare hidden text conditioning for each regional prompt using the same Krea2 text path as the model:
   - unpack regional conditioning context
   - run `txtfusion`
   - run `txtmlp`
   - match dtype/device/batch to the current forward
4. Hook Krea2 transformer blocks (`blocks.*`) or their attention output boundary.
5. For each regional prompt, compute a regional text-conditioning delta against global text conditioning. Initial implementation will use a pooled text hidden delta per region, broadcast only onto the matching image-token mask.
6. Add the regional prompt delta to hidden states only in `hidden[:, image_start:image_end]` where the soft region token mask is active.
7. Leave text tokens and padding tokens unchanged.
8. Log per-layer regional prompt delta stats when debug is enabled.

This is intentionally inside the model forward/transformer layer execution, not after sampler CFG prediction.

## Regional LoRA Application

The existing model-side LoRA hook remains the backbone but will be tightened:

1. LoRA residuals are computed at matched linear layers.
2. The residual mask is constructed with `image_start:image_end` from runtime layout.
3. The residual is multiplied by the soft `KREA_REGION` token mask.
4. Text tokens receive no regional LoRA by default.
5. Padding tokens always receive zero regional LoRA.
6. Runtime token masks are resized from the canonical pixel mask when runtime image grid differs from the static region token grid.

## Delta Measurement And Token Flags

For each region and layer update:

1. Collect selected measurements:
   - direct LoRA residual
   - hidden-state delta
   - final prediction delta only if enabled as a diagnostic source
2. Normalize each source using the selected normalization mode.
3. Combine selected sources by averaging normalized scores.
4. Restrict scoring to `image_start:image_end`.
5. Multiply scores by the region token mask.
6. Flag image tokens whose combined score exceeds `modified_threshold`.
7. Preserve/update flags using existing retention controls:
   - `sticky`
   - `decay`
   - `instant`

The state will store both the soft influence scores and the boolean modified-token mask.

## Attention Isolation

Attention isolation will be implemented through the existing optimized attention override path, but the bias will use the captured text/image/padding layout.

Masks:

- `image_tokens`: true only for `image_start:image_end`
- `modified_tokens`: true only for flagged image tokens
- `unmodified_image_tokens`: image tokens not flagged
- padding tokens: excluded from all regional masks

Controls:

- `attention_isolation_mode=none`: no unmodified-to-modified bias
- `penalize`: subtract `attention_isolation_strength` from unmodified image queries attending to modified image keys
- `forbid`: subtract a large block value from unmodified image queries attending to modified image keys
- `modified_outward_mode=none`: modified tokens may attend outward
- `modified_outward_mode=penalize/forbid`: optionally penalize/block modified image queries attending to unmodified image keys
- cross-region bias remains controlled by `cross_lora_mode`

## Debug Output

Debug reporting will include:

- `region_id`
- pixel bbox and normalized bbox
- `text_len`, `image_len`, `pad_len`
- `image_start`, `image_end`
- bbox-to-token mask stats and nonzero range
- modified-token mask count and nonzero range
- attention-isolation mask stats
- per-layer direct delta stats
- per-layer hidden-state delta stats
- optional final prediction delta stats when enabled

Implementation details:

- `debug_logging=True` logs runtime per-forward/per-layer details to ComfyUI logs.
- Existing debug nodes will continue to preview static region masks.
- A new or extended debug report function will format the latest runtime state when available. If ComfyUI execution order cannot guarantee a post-sampler debug node, runtime details will still be available in logs.

## Files To Change

Expected files:

- `krea_region_lora/types.py`
  - add isolation mode type
  - add final prediction diagnostic measurement source
  - extend stack fields

- `krea_region_lora/tracking.py`
  - own runtime layout
  - own modified-token retention state
  - build attention isolation masks from image span, not sequence tail
  - collect debug stats

- `krea_region_lora/layer_injection.py`
  - capture Krea2 layout from model forward args
  - keep regional LoRA residuals inside image span only
  - add per-layer measurement reporting

- `krea_region_lora/conditioning.py`
  - rename current CFG implementation to sampler fallback
  - add transformer-layer conditioning implementation
  - prepare regional text hidden deltas for hooks

- `nodes.py`
  - expose conditioning mode selector
  - expose explicit attention isolation/outward controls
  - update debug report text

- `tests/test_core.py`
  - cover text/image/padding layout
  - cover no tail masking after padding
  - cover isolation modes
  - cover source combination/retention behavior
  - cover sampler fallback being opt-in

- `README.md` and workflow example
  - document default transformer hook mode
  - mark sampler fallback as experimental
  - update example workflow to use default mode

## Validation Plan

Run:

```bash
python3 run_tests.py
python3 -m py_compile nodes.py krea_region_lora/*.py tests/test_core.py
```

Source checks:

- no `seq_len - img_len` image-token placement except for computing `pad_len = seq_len - image_end`
- no sampler CFG regional prompt path unless `conditioning_mode == "sampler_delta_conditioning"`
- tests prove padding tokens remain unmasked

Manual workflow check:

- Load `regional_prompt_conditioning_example.json`
- Confirm `KreaRegionalConditioningApply` defaults to transformer hook mode
- Confirm region 1 and region 2 feed both regional prompt and regional LoRA nodes from the same `KREA_REGION` outputs
