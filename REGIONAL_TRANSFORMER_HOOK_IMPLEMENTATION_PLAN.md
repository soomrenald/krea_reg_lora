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

Warning: the pooled text hidden delta path is acceptable only as v1. It is transformer-internal and better than sampler CFG patching, but it is not equivalent to true regional prompt-token attention control. If pooled text deltas do not visibly affect subject placement, prioritize implementing regional prompt/image self-attention bias before tuning LoRA strengths.

Implementation steps:

1. `KreaRegionalPrompt` continues to encode each regional prompt with CLIP.
2. `KreaRegionalConditioningApply` stores the regional conditioning stack in the model patch wrapper.
3. In the diffusion-model wrapper, prepare hidden text conditioning for each regional prompt using the same Krea2 text path as the model:
   - unpack regional conditioning context
   - run `txtfusion`
   - run `txtmlp`
   - match dtype/device/batch to the current forward
4. Before implementing pooled text deltas, verify the exact insertion point available in the Krea2 block code and record whether the hook is:
   - pre-attention
   - post-attention
   - post-MLP
   - block output
5. Expose the chosen insertion point as debug/config metadata if practical. If a user-facing selector is added, its default must be `post_attention` or `block_output`, not an arbitrary internal boundary.
6. Hook Krea2 transformer blocks (`blocks.*`) at the verified attention output boundary or block output boundary. Do not proceed with pooled text deltas until this boundary is confirmed in code.
7. For each regional prompt, compute a regional text-conditioning delta against global text conditioning. Initial implementation will use a pooled text hidden delta per region, broadcast only onto the matching image-token mask.
8. Add the regional prompt delta to hidden states only in `hidden[:, image_start:image_end]` where the soft region token mask is active.
9. Leave text tokens and padding tokens unchanged.
10. Log per-layer regional prompt delta stats when debug is enabled.

This is intentionally inside the model forward/transformer layer execution, not after sampler CFG prediction.

Hard implementation constraint: do not proceed with broad transformer patching until the exact Krea2 class/function/boundary being hooked is identified and logged. The implementation report/debug output must state the exact hook target:

- class name
- function name
- whether hook is pre-attention, post-attention, post-MLP, or block output
- hidden tensor shape at that boundary
- whether the tensor includes text + image + padding tokens

The pooled-delta implementation is only the first conditioning path. The hook architecture must also reserve a path for true self-attention bias between regional prompt text tokens and regional image tokens. This means regional prompt token ranges must be tracked as first-class runtime metadata, not collapsed permanently into a pooled vector.

Regional prompt attention bias requirements:

1. Region image-token queries can attend to their matching regional prompt-token keys.
2. Non-region image-token queries are penalized or forbidden from attending to those regional prompt-token keys.
3. Regional prompt tokens do not globally influence all image tokens unless `outside_strength > 0`.

Future/parallel attention-bias requirements:

- image tokens inside a region may attend to that region's prompt tokens
- image tokens outside the region are penalized/forbidden from attending to that region's prompt tokens
- regional prompt tokens must not globally influence all image tokens unless `outside_strength > 0`
- this must be separate from LoRA modified-token isolation
- regional prompt token ranges must remain first-class runtime metadata, not be permanently collapsed into a pooled vector

The implementation should preserve enough metadata to add this attention-bias path without rewriting regional prompt conditioning after pooled text deltas land.

Tuning note: do not try to compensate for weak regional prompt placement by raising LoRA strength or delta strength first. If placement remains weak with correct masks, implement or enable regional prompt/image attention bias before escalating LoRA values.

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
- `regional_prompt_tokens`: true only for runtime text-token ranges belonging to a regional prompt, when regional prompt token ranges are available
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
- regional prompt/image attention bias will be represented separately from LoRA modified-token isolation so text conditioning leakage can be controlled independently

Regional prompt attention-bias behavior:

- Matching region image queries to matching regional prompt keys: allowed.
- Non-region image queries to regional prompt keys: penalized or forbidden according to the regional prompt isolation controls.
- Regional prompt queries/keys must not create a global text-to-image influence path unless the region explicitly sets `outside_strength > 0`.
- Padding rows and columns remain unaffected except for whatever base model mask already applies.

## Debug Output

Debug reporting will include:

- `region_id`
- pixel bbox and normalized bbox
- `text_len`, `image_len`, `pad_len`
- `image_start`, `image_end`
- bbox-to-token mask stats and nonzero range
- actual runtime token mask stats and nonzero range for each forward, based on `image_start:image_end`, not only the static bbox preview
- regional prompt token ranges when available
- modified-token mask count and nonzero range
- modified-token flags proving flagged tokens are inside the image span
- regional prompt and regional LoRA same-mask verification fields:
  - `regional_prompt.region_id`
  - `regional_lora.region_id`
  - regional prompt token-mask nonzero range
  - regional LoRA token-mask nonzero range
  - mask shape
  - mask min/max/mean
  - optional mask hash or checksum
- attention-isolation mask stats
- regional prompt/image attention-bias mask stats when enabled
- per-layer direct delta stats
- per-layer hidden-state delta stats
- optional final prediction delta stats when enabled

Implementation details:

- `debug_logging=True` logs runtime per-forward/per-layer details to ComfyUI logs.
- Existing debug nodes will continue to preview static region masks.
- Add either a debug image or a log/report view that shows the actual runtime token mask after layout capture. This must reflect the runtime token span and padding, not only the static bbox-to-token preview.
- A new or extended debug report function will format the latest runtime state when available. If ComfyUI execution order cannot guarantee a post-sampler debug node, runtime details will still be available in logs.
- If regional prompt and LoRA are fed from the same `KREA_REGION`, their runtime mask shape and nonzero token range should match. If not, log a warning.

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
  - add a padding case proving text tokens are unchanged by regional LoRA
  - add a padding case proving padding tokens are unchanged
  - add a padding case proving only `image_start:image_end` receives the regional mask
  - add a padding case proving modified-token flags are only inside the image span
  - add an all-zero region mask case proving regional prompt and regional LoRA cause zero changed image-token hidden states
  - add an all-one image mask case proving regional prompt and regional LoRA can affect all image tokens and no text/padding tokens
  - add a left-half image mask case proving only left-half image tokens receive nonzero regional prompt/LoRA deltas
  - add a padding case proving padding tokens remain unchanged
  - add a text-token case proving text tokens remain unchanged by regional LoRA
  - add a modified-token case proving flags cannot appear outside `image_start:image_end`
  - cover isolation modes
  - cover regional prompt/image attention-bias mask construction when regional prompt token ranges are available
  - cover same-`KREA_REGION` prompt/LoRA runtime mask alignment and warning behavior on mismatch
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

- no regional prompt behavior runs through sampler CFG unless `conditioning_mode == "sampler_delta_conditioning"`
- no image-token insertion uses `seq_len - img_len` except for image-only/no-padding cases or `pad_len` calculations
- no static debug bbox mask is used as proof of actual runtime injection alignment
- tests prove padding tokens remain unmasked
- tests prove text tokens are unchanged by regional LoRA
- tests prove modified-token flags cannot appear outside `image_start:image_end`
- debug output proves actual runtime token masks are available, not only static bbox previews

Manual workflow check:

- Load `regional_prompt_conditioning_example.json`
- Confirm `KreaRegionalConditioningApply` defaults to transformer hook mode
- Confirm the example workflow shows region 1 and region 2 feed both `KreaRegionalPrompt` and `KreaRegionalLoRA` from the exact same `KREA_REGION` output
- Confirm debug output shows the verified hook insertion point and actual runtime token mask
- Confirm debug output shows prompt and LoRA runtime masks align for each region
