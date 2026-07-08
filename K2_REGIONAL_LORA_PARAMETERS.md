# K2 Regional LoRA Parameters And Tuning

This workflow has two related goals:

1. Keep character LoRA effects inside the selected bbox regions.
2. Make the regional LoRA effect strong enough to be visible.

The current practical path is the sampler's `layer_injection` fallback. It runs a base sample, then a regional sample with masked LoRA layer injection, then optionally pins everything outside the union mask back to the base result.

## Optional Standalone Layer Apply Node

`K2 Regional Layer LoRA Apply` is an alternate way to use the same layer-injection fallback.

Use either:

- `K2 Regional Attention LoRA Sampler`, recommended. It handles the base pass, regional pass, outside latent pinning, and returns `base_samples`, `regional_samples`, `union_mask`, and `debug_info`.
- `K2 Regional Layer LoRA Apply` before a normal Comfy sampler. This only patches the model. It does not run the base pass, does not return a base latent, and does not do final outside-region pinning by itself.

Do not use both in the same sampler path. If `Layer LoRA Apply` feeds the regional sampler, you can double-apply or confuse the fallback path.

The standalone node is useful for diagnostics because its `report` output tells you how many LoRA layers were matched and patched.

## K2 BBox To Regional Mask

Converts a KJ bbox into a pixel mask, latent mask, token mask, and `K2REGION` object.

### Inputs

- `bboxes`: KJ `BOUNDING_BOX` input. Use this with `Ideogram 4 Prompt Builder KJ`.
- `kj_bboxes`: legacy `BBOX` input.
- `latent`: optional latent input. When connected, image size and batch can be inferred from the latent.

### Parameters

- `width`, `height`: image canvas size. If `latent` is connected, latent-derived size usually wins.
- `bbox_format`: `xywh` or `xyxy`.
  - Use `xywh` for KJ bbox output unless you know the source emits corners.
- `bbox_index`: which KJ box to use.
  - KJ displays boxes as `1`, `2`, `3`.
  - This node follows that numbering.
  - `0` is accepted as an alias for the first box for older workflows.
- `grow_px`: expands or shrinks the bbox before mask creation.
  - Positive values make the LoRA region larger.
  - Negative values tighten it.
- `feather_px`: softens the inside edge of the mask.
  - Higher values reduce harsh seams but can weaken edges.
- `snap_to_krea_token_grid`: expands bbox edges to Krea token boundaries.
  - Usually leave on.
- `batch_mode`: mask batch behavior.
  - `repeat`: same mask for every batch item.
  - `per_batch`: preserve batch-sized masks.
  - `single`: one mask.

### Outputs

- `region_mask`: previewable pixel mask.
- `region`: connect to `K2 Regional Character LoRA`.
- `debug_bbox_image`: quick visual check that the selected bbox is correct.

## K2 Regional Character LoRA

Binds one LoRA file to one region.

### Inputs

- `region`: from `K2 BBox To Regional Mask`.
- `positive`: regional positive conditioning metadata.
- `negative`: regional negative conditioning metadata.

Important: in the current layer-injection fallback, regional `positive` and `negative` are stored but not used for separate per-region text conditioning. Put all character trigger words in the sampler's base positive prompt.

### Parameters

- `lora_name`: LoRA file from Comfy's `loras` folder.
- `lora_strength`: model-side LoRA strength multiplier.
- `delta_strength`: extra regional delta multiplier.
  - Effective strength is roughly `lora_strength * delta_strength * LoRA internal scale`.
- `start_percent`: denoising fraction where this region starts applying.
  - Lower means earlier identity influence.
- `end_percent`: denoising fraction where this region stops applying.
  - Higher means later identity/detail influence.
- `enabled`: turns this regional LoRA on/off.
- `attention_only_filter`: used by the strict adapter path. In fallback mode, use sampler `layer_injection_targets` instead.
- `ignore_text_encoder_lora`: skips text-encoder LoRA weights.
  - Usually leave on for containment.
  - If your LoRA relies heavily on text-encoder weights, this can reduce apparent strength.

## K2 Regional LoRA Stack 3

Combines up to three `K2REGIONAL_LORA` bindings.

### Inputs

- `regional_lora_1`: required.
- `regional_lora_2`: optional.
- `regional_lora_3`: optional.

### Parameters

- `overlap_mode`: behavior where region masks overlap.
  - `normalize`: blends overlapping LoRA deltas so total strength does not explode.
  - `priority_1`: first region wins overlaps.
  - `priority_3`: later region wins overlaps.
  - `add_clamped`: adds overlap strength, then clamps.

For separated character boxes, `normalize` is a good default. For overlapping faces/bodies, priority modes are often easier to reason about.

## K2 Regional Attention LoRA Sampler

Runs the denoising process with regional LoRA behavior.

### Inputs

- `model`: base Krea diffusion model.
- `positive`: global/base positive conditioning.
- `negative`: global/base negative conditioning.
- `latent_image`: starting latent.
- `regional_lora_stack`: from `K2 Regional LoRA Stack 3`.

### Basic Sampler Parameters

- `seed`: random seed.
- `steps`: denoising steps.
- `cfg`: classifier-free guidance scale.
- `sampler_name`: Comfy sampler.
- `scheduler`: Comfy scheduler.
- `denoise`: normal Comfy denoise amount.

### Regional Parameters

- `execution_mode`:
  - `auto`: use strict adapter if available, otherwise fallback to layer injection.
  - `strict_adapter`: only use an adapter model exposing `k2_regional_velocity_predictor`; otherwise return base samples.
  - `layer_injection`: force fallback mode.
  - `branch_lora_composite`: diagnostic/reliable mode. Runs one normal Comfy LoRA branch per enabled regional LoRA, then composites each branch's masked latent delta into the base latent. This is slower, but it uses Comfy's proven LoRA loading path and is the first mode to try when `layer_injection` shows no visible LoRA effect.
- `layer_injection_targets`:
  - `attn_out_mlp`: default containment-focused choice. Patches attention output and MLP/writeback layers.
  - `attention_only`: patches attention projections, including Q/K/V/gate and output-like attention layers. This can increase effect but risks more mixing.
  - `all_matched_linears`: strongest diagnostic mode. Patches every matched linear layer from the LoRA. Highest leakage risk.
- `layer_outside_strength`:
  - LoRA delta strength outside masks in fallback mode.
  - `0.0` is containment mode.
  - Small values like `0.03` to `0.10` are useful only for diagnostics or if you accept leakage.
- `layer_text_token_strength`:
  - LoRA delta strength on text tokens in fallback mode.
  - `0.0` is containment mode.
  - Try `0.05` to `0.25` if the LoRA has no visible effect and seems to need trigger/text interaction.
- `pin_outside_regions`:
  - Used by the strict adapter path for per-step outside-region pinning.
- `final_latent_pin`:
  - In fallback mode, pins final regional latent outside the union mask back to the base latent.
  - Leave on for final images.
  - Turn off temporarily to diagnose whether the regional LoRA is doing anything globally.
- `post_decode_safe_mode`:
  - Reserved in the current implementation. The decode composite node performs the post-decode compositing step.
- `debug_return_base_latent`:
  - Returns `base_samples` for comparison/compositing.

### Outputs

- `samples`: regional result.
- `base_samples`: no-LoRA base result.
- `union_mask`: combined pixel mask for all regions.
- `debug_info`: text report. Connect this to a text display node if available.

## K2 Regional Decode Composite

Decodes regional and base latents, then composites decoded images by `union_mask`.

### Inputs

- `vae`: Krea VAE.
- `regional_samples`: sampler `samples`.
- `base_samples`: sampler `base_samples`.
- `union_mask`: sampler `union_mask`.

### Parameters

- `feather_px`: post-decode compositing feather.
  - Higher values soften the transition from regional image to base image.
  - Too high can visually dilute edge details near a character.

### Output

- `IMAGE`: normal Comfy image batch for preview/save.

## Why A Workflow Can Show Zero LoRA Effect

Check these in order:

1. The base positive prompt may not contain the LoRA trigger tokens.
   - In fallback mode, put all character triggers in the base positive prompt.
   - The regional positive prompt inputs are not separate regional text prompts yet.
2. The bbox mask may not cover the subject.
   - Preview `debug_bbox_image`, `region_mask`, or `union_mask`.
   - If the box is off, confirm KJ bbox numbering and bbox format.
3. The LoRA may not match Krea2 model layer names.
   - Connect `debug_info` to a text display node.
   - Look for `matched_layers`.
   - If `matched_layers=0`, the LoRA loaded but did not patch any usable model layers.
4. Strength may be too low for masked layer injection.
   - Increase `lora_strength` and `delta_strength`.
5. Containment settings may be too strict for a weak or text-heavy LoRA.
   - Temporarily increase `layer_text_token_strength`.
   - Temporarily use `all_matched_linears`.
6. Final pinning may be hiding a mask problem.
   - Temporarily set `final_latent_pin=false`.
   - If the LoRA appears globally with pinning off, the LoRA works but the mask is wrong, too small, or not covering visible identity regions.
7. The experimental layer-injection hooks may not be affecting this exact Krea model build.
   - Switch `execution_mode` to `branch_lora_composite`.
   - Set each character node `attention_only_filter=false`.
   - This should produce visible LoRA effect if the LoRA works globally.

## Branch LoRA Composite Mode

`branch_lora_composite` is the most reliable fallback mode.

It runs:

- one base sampler pass,
- plus one normal LoRA sampler pass per enabled regional LoRA.

So two character LoRAs cost three total denoise passes. Three character LoRAs cost four total denoise passes.

For each character branch:

1. The LoRA is loaded through Comfy's normal model LoRA path.
2. The branch is sampled using that LoRA.
3. The node computes `branch_latent - base_latent`.
4. Only the masked regional delta is composited back into the base latent.

This mode is less elegant than true per-layer regional injection, but it is much harder for the LoRA effect to disappear silently. It also lets each `K2 Regional Character LoRA` node's `positive` and `negative` conditioning be used for that branch.

Recommended diagnostic settings:

- Sampler:
  - `execution_mode=branch_lora_composite`
- Character LoRA:
  - `lora_strength=1.5`
  - `delta_strength=1.25`
  - `attention_only_filter=false`
  - `ignore_text_encoder_lora=true`

If this still shows zero LoRA effect, test the same LoRA with a normal global Comfy LoRA loader. The likely issue is then trigger text, LoRA/model compatibility, or the LoRA file itself.

## Tuning Strategy To Increase LoRA Effect

Start with a diagnostic run, then tighten containment again.

### Step 1: Prove The LoRA Works At All

Run the same LoRA with Comfy's normal global LoRA loader on the same Krea model.

If the LoRA is weak globally, it will be weaker regionally.

### Step 2: Prove The Region Mask Is Correct

Use:

- `debug_bbox_image` from each `K2 BBox To Regional Mask`.
- `union_mask` from the sampler.

The mask should cover the whole face/head/body area that needs identity. For characters, a tiny face-only box often has less effect than a head-and-upper-body box.

Suggested changes:

- Increase `grow_px` to `32`, `64`, or `96`.
- Keep `snap_to_krea_token_grid=true`.
- Try `feather_px=16` in the bbox node if the mask is too soft.

### Step 3: Make The Prompt Trigger Unambiguous

Put the triggers in the base positive prompt:

```text
lface woman in the left bbox, sface man in the right bbox, two separate people, distinct faces
```

Use the actual trigger tokens your LoRAs were trained with. If the trigger is not present in the base prompt, fallback mode may have little to steer.

### Step 4: Increase Per-LoRA Strength

On each `K2 Regional Character LoRA`:

- Try `lora_strength=1.25`, `delta_strength=1.25`.
- Then `1.5` and `1.5`.
- For weak LoRAs, try `2.0` and `1.5`.

Avoid changing everything at once. `lora_strength` affects the loaded LoRA delta; `delta_strength` is the regional multiplier after masking.

### Step 5: Apply Earlier And Longer

For identity LoRAs, try:

- `start_percent=0.00`
- `end_percent=1.00`

If the face becomes overcooked, back off to:

- `start_percent=0.05`
- `end_percent=0.90`

### Step 6: Broaden Layer Targets

On the sampler:

1. Start: `layer_injection_targets=attn_out_mlp`.
2. If effect is weak: try `all_matched_linears`.
3. If effect appears but leaks too much: return to `attn_out_mlp` and increase LoRA strength instead.

`all_matched_linears` is best treated as a diagnostic maximum-strength mode.

### Step 7: Let Text Tokens Participate Slightly

If the LoRA relies on trigger-token interaction:

- Try `layer_text_token_strength=0.05`.
- Then `0.10`.
- Then `0.20`.

This can increase identity strength, but it is less region-pure because text tokens are shared across the whole image.

### Step 8: Use Outside Strength Only As A Diagnostic

Temporarily try:

- `layer_outside_strength=0.05`

If the LoRA suddenly becomes visible, the LoRA is loaded and active, but pure masked injection is too constrained. Return to `0.0` for containment and instead increase mask size, text token strength, or layer targets.

### Step 9: Temporarily Disable Final Pinning

Set:

- `final_latent_pin=false`

If the LoRA becomes visible only with pinning disabled, the issue is probably mask coverage or bbox selection, not LoRA loading.

Turn `final_latent_pin=true` again for final regional outputs.

## Avoiding Visible BBox Overlays In The Output

Use the KJ node as a bbox/size source, not as the final generation prompt source.

Connect:

- `Ideogram 4 Prompt Builder KJ:bboxes` to each `K2 BBox To Regional Mask:bboxes`.
- `Ideogram 4 Prompt Builder KJ:width` and `height` to mask/latent size widgets if desired.

Do not connect:

- `Ideogram 4 Prompt Builder KJ:prompt` to `CLIP Text Encode:text`.
- `Ideogram 4 Prompt Builder KJ:preview` to the final save/preview image.

The KJ preview intentionally draws visible rectangles, labels, and coordinates. The KJ prompt output may also include bbox labels or coordinate-like text. If that string is sent into CLIP, the model can draw boxes, labels, and numbers into the generated image.

Use a clean base positive prompt instead:

```text
lface woman standing on the left side of the image, sface man standing on the right side of the image, two separate people, full body, neutral indoor studio, natural light, distinct non-blended faces
```

Add bbox overlay artifacts to the negative prompt:

```text
bounding boxes, white rectangles, coordinate labels, text overlay, annotations, labeled diagram
```

## Suggested Strong Diagnostic Settings

Use these only to determine whether the LoRA can show up:

- Character LoRA:
  - `lora_strength=1.75`
  - `delta_strength=1.50`
  - `start_percent=0.00`
  - `end_percent=1.00`
- BBox mask:
  - `grow_px=64`
  - `feather_px=16`
- Sampler:
  - `execution_mode=layer_injection`
  - `layer_injection_targets=all_matched_linears`
  - `layer_text_token_strength=0.10`
  - `layer_outside_strength=0.00`
  - `final_latent_pin=false` for one diagnostic run only

If that shows an effect, tighten back toward:

- `layer_injection_targets=attn_out_mlp`
- `layer_text_token_strength=0.00`
- `final_latent_pin=true`
- lower LoRA strengths until identity remains but artifacts/leakage reduce.

## Suggested Contained Production Settings

- Character LoRA:
  - `lora_strength=1.25` to `1.75`
  - `delta_strength=1.00` to `1.50`
  - `start_percent=0.00` to `0.10`
  - `end_percent=0.90` to `1.00`
- BBox mask:
  - `grow_px=32` to `96`
  - `feather_px=16` to `48`
- Sampler:
  - `execution_mode=auto`
  - `layer_injection_targets=attn_out_mlp`
  - `layer_outside_strength=0.00`
  - `layer_text_token_strength=0.00` to `0.10`
  - `final_latent_pin=true`
- Decode composite:
  - `feather_px=16` to `48`
