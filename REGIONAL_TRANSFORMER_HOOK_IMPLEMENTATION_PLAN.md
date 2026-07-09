# Regional Transformer Hook Implementation Plan

## Objective

Replace the current default regional prompt implementation that patches sampler CFG output with transformer attention/model hooks. The CFG path will remain only as an explicitly selected fallback mode named `sampler_delta_conditioning`, disabled by default. The default regional prompt implementation must use regional prompt-token to regional image-token self-attention control inside Krea2 transformer attention.

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

- `conditioning_mode`: `transformer_attention_bias` default
- `transformer_attention_bias`: default actual implementation using prompt-token/image-token attention bias
- `sampler_delta_conditioning`: old CFG/sampler fallback, explicitly experimental and opt-in only
- `disabled`: passes model through while preserving debug stack outputs

Do not add `transformer_layer_conditioning` as a pooled-delta default. If a pooled hidden-state delta path is retained for diagnostics, it must be explicitly named `pooled_hidden_delta_experimental` and disabled by default.

The existing sampler-CFG code path will be renamed in code and report text to `sampler_delta_conditioning` so it is not confused with the default implementation.

### `KreaRegionalPrompt`

`KreaRegionalPrompt` must output:

- regional conditioning data
- `region_id`
- token range metadata when available
- `strength`
- `outside_strength`
- `prompt_attention_mode`: `none`, `penalize`, `forbid`
- `prompt_attention_strength`: float used by `penalize`

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

Krea2 is a single-stream transformer: text tokens and image tokens are concatenated into the same self-attention sequence. Therefore regional prompt conditioning must be implemented by controlling self-attention between regional prompt text tokens and image tokens, not by sampler CFG patching and not by pooled text-delta broadcasting.

Required behavior:

1. Track global prompt token range, regional prompt token ranges, image token range, and padding range at runtime.
2. Use Krea2 layout:
   - text tokens first
   - image tokens second
   - padding after image tokens
   - `image_start = text_len`
   - `image_end = text_len + image_len`
3. Regional prompt tokens must remain first-class tokens/ranges. Do not permanently collapse them into a pooled vector for the default implementation.
4. For each region, build an attention-bias mask so:
   - image-token queries inside the region may attend to that region's prompt-token keys
   - image-token queries outside the region are penalized or forbidden from attending to that region's prompt-token keys
   - regional prompt tokens do not globally influence unrelated image tokens unless `outside_strength > 0`
   - padding rows/columns remain governed by the base model mask and receive no regional behavior
5. Regional prompt attention bias must be separate from LoRA modified-token attention isolation.
6. Regional prompt attention bias must use the same `KREA_REGION` runtime token mask as the matching regional LoRA.
7. The implementation must happen inside transformer attention/model hooks, not by running extra denoiser predictions and masking prediction deltas.
8. `sampler_delta_conditioning` may remain only as an explicitly selected experimental fallback, disabled by default.

Default implementation requirement:

For each regional prompt, preserve its token range and construct per-layer attention bias between that regional prompt token range and the matching region's image-token mask. The default implementation must use attention bias/token routing, not pooled text hidden-state deltas.

Important implementation instruction: do not implement the pooled text hidden delta approach as v1. Implement the actual regional prompt-token/image-token self-attention bias path first. If the attention hook proves impossible after code inspection, stop and report the blocker instead of substituting a pooled-delta or CFG approximation.

Attention-bias implementation requirements:

1. Identify the exact Krea2 attention implementation being hooked before modifying behavior.
2. Log:
   - class name
   - function name
   - hook boundary
   - hidden shape
   - attention shape
   - active attention backend/path used for each hooked attention call, because additive attention-bias support may differ between torch SDPA, xformers, sage attention, and custom optimized attention
   - `text_len`
   - `image_len`
   - `pad_len`
   - `image_start`/`image_end`
3. Build an additive attention bias tensor compatible with the attention implementation.
4. For each regional prompt:
   - let `P_r` = regional prompt token key positions
   - let `I_r` = image token query positions inside region `r`
   - let `I_not_r` = image token query positions outside region `r`
   - allow `I_r -> P_r`
   - penalize/forbid `I_not_r -> P_r`
   - optionally allow small `outside_strength` instead of full blocking
5. Do not block global prompt tokens unless explicitly requested.
6. Do not apply regional prompt bias to padding tokens.
7. Do not apply regional prompt bias to unrelated regions except through explicit cross-region controls.
8. Support multiple regions and avoid region prompt leakage across boxes.

Tuning note: do not try to compensate for weak regional prompt placement by raising LoRA strength or delta strength first. If placement remains weak with correct masks, inspect and correct regional prompt/image attention bias before escalating LoRA values.

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

LoRA modified-token isolation remains separate:

- `direct_delta` / `hidden_state_delta` measure LoRA effects
- normalized scores flag modified image tokens
- unmodified image-token queries can be penalized/forbidden from attending to modified image-token keys
- modified outward attention is separately controlled
- this must not be confused with regional prompt-token attention bias

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
  - prompt `region_id`
  - LoRA `region_id`
  - image token mask shape
  - prompt attention mask nonzero query/key ranges
  - LoRA mask nonzero range
  - regional prompt token-mask nonzero range
  - regional LoRA token-mask nonzero range
  - mask shape
  - mask min/max/mean
  - optional mask hash or checksum
  - whether prompt mask and LoRA mask align
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
- If the prompt and LoRA are fed from the same `KREA_REGION` but their runtime masks differ, log a warning.

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
  - add transformer attention-bias conditioning implementation
  - preserve regional prompt token ranges for hooks
  - keep any pooled hidden delta path only as `pooled_hidden_delta_experimental`, disabled by default

- `nodes.py`
  - expose conditioning mode selector
  - expose explicit attention isolation/outward controls
  - update debug report text

- `tests/test_core.py`
  - cover attention bias construction:
    - region image tokens may attend to matching regional prompt tokens
    - non-region image tokens are penalized/forbidden from attending to regional prompt tokens
    - padding tokens are unaffected
    - unrelated regional prompt tokens do not leak into other regions
  - cover layout:
    - text tokens before image tokens
    - `image_start = text_len`
    - `image_end = text_len + image_len`
    - no `seq_len - img_len` tail assumption except image-only/no-padding case or `pad_len` calculation
  - cover same-mask alignment:
    - `KreaRegionalPrompt` and `KreaRegionalLoRA` fed from same `KREA_REGION` produce matching runtime region masks
  - cover region masks:
    - all-zero mask produces no regional prompt attention bias and no LoRA effect
    - all-one image mask affects all image tokens and no text/padding tokens
    - left-half mask affects only left-half image tokens
  - cover isolation:
    - LoRA modified-token flags cannot appear outside `image_start:image_end`
    - `attention_isolation_mode=forbid` creates large negative bias from unmodified image queries to modified image keys
    - `modified_outward_mode` controls modified queries attending outward separately
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

- no default regional prompt path uses sampler CFG
- no default regional prompt path uses pooled text hidden delta
- no image-token insertion uses `seq_len - img_len` except for image-only/no-padding cases or `pad_len` calculations
- regional prompt attention bias is active only when `conditioning_mode == "transformer_attention_bias"`
- `sampler_delta_conditioning` is opt-in and clearly marked experimental
- no static debug bbox mask is used as proof of actual runtime injection alignment
- tests prove padding tokens remain unmasked
- tests prove text tokens are unchanged by regional LoRA
- tests prove modified-token flags cannot appear outside `image_start:image_end`
- debug output proves actual runtime token masks are available, not only static bbox previews

Manual workflow acceptance:

1. Load `regional_prompt_conditioning_example.json`.
2. Confirm `KreaRegionalConditioningApply` defaults to `transformer_attention_bias`.
3. Confirm region 1 and region 2 each feed both:
   - `KreaRegionalPrompt`
   - `KreaRegionalLoRA`
   from the exact same `KREA_REGION` output.
4. Run with regional prompt enabled and regional LoRA disabled.
5. Run with regional prompt disabled and regional LoRA enabled.
6. Run with both enabled.
7. Debug output must show each subsystem independently uses the same region mask.
8. Debug output must show actual runtime attention-bias query/key ranges, not only static bbox previews.
