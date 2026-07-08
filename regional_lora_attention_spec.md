# Regional LoRA Delta Tracking and Asymmetric Attention Masking Spec

## 1. Purpose

Add regional LoRA support to a Krea2-style transformer diffusion model while reducing LoRA concept leakage outside the intended region.

The system should:

- Apply each LoRA only to user-selected spatial regions.
- Track which image tokens are actually modified by each LoRA.
- Allow users to choose how LoRA modification deltas are measured.
- Convert measured deltas into per-token "modified" flags using user-controlled thresholds.
- Prevent unmodified tokens from attending to LoRA-modified tokens.
- Preserve background and scene continuity by avoiding crude whole-box isolation.
- Support multiple regions and multiple LoRAs.

## 2. Core Concept

The model sequence is assumed to contain text tokens and image latent tokens:

```text
[text tokens][image tokens]
```

Each image token corresponds to a spatial patch of the latent image grid. A user-provided pixel-space bounding box is converted to an image-token mask.

For each region-specific LoRA:

```text
LoRA delta may be nonzero only for image tokens inside the selected region.
```

Then the system tracks which tokens were materially changed by the LoRA, not merely which tokens are inside the box.

## 3. High-Level Algorithm

For each denoising step:

```text
for each transformer block:
    1. Apply regional LoRA delta only to image tokens inside each LoRA region.
    2. Measure LoRA modification magnitude per token.
    3. Normalize measurement values.
    4. Average selected measurement sources if more than one is enabled.
    5. Compare averaged score to user threshold.
    6. Update per-LoRA modified-token flags.
    7. Apply asymmetric attention mask:
         tokens not modified by that LoRA cannot attend to tokens modified by that LoRA.
    8. Continue forward pass.
```

## 4. Regional LoRA Application

Given:

- `x`: transformer hidden states, shape `[batch, seq_len, hidden_dim]`
- `txt_len`: number of text tokens
- `img_len`: number of image tokens
- `region_mask`: Boolean mask over image tokens, shape `[img_len]`
- `lora_delta`: LoRA output delta, shape `[batch, seq_len, hidden_dim]`

Build full sequence mask:

```python
full_region_mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
full_region_mask[txt_len:txt_len + img_len] = region_mask
```

Apply LoRA regionally:

```python
x = base_output + full_region_mask[None, :, None] * lora_delta
```

Only tokens inside the region can receive nonzero direct LoRA delta.

## 5. Delta Measurement Sources

Users must be able to select one or more delta measurement sources.

If multiple are selected, each source is normalized independently, then the normalized values are averaged to produce a final per-token modification score.

### 5.1 Per-LoRA-Layer Direct Delta

Measures the direct output of each LoRA-wrapped layer.

```python
delta = lora_delta
score = norm(delta, dim=-1)
```

Pros:

- Cheapest.
- Precisely measures where LoRA was directly injected.
- Easy to implement inside LoRA wrappers.

Cons:

- Does not capture propagated LoRA influence after attention mixing.
- May miss tokens indirectly affected by earlier LoRA-modified tokens.

Recommended as the default first implementation.

### 5.2 Per-Transformer-Block Output Delta

Compares block output with and without regional LoRA effect.

```python
delta = block_output_regional - block_output_base
score = norm(delta, dim=-1)
```

Pros:

- Captures accumulated effect across attention and MLP within the block.
- More semantically meaningful than direct LoRA delta alone.

Cons:

- Requires a base/reference block output.
- More expensive than direct LoRA-layer tracking.

### 5.3 Attention-Propagated Influence Score

Tracks how much each token attends to already modified tokens.

For attention matrix:

```text
attention[query_token, key_token]
```

If a query token attends to modified key/value tokens, it receives propagated influence.

Example:

```python
attn_avg = attention.mean(dim=1)          # average heads, [batch, query, key]
propagated = attn_avg @ modified_scores   # [batch, query]
```

Pros:

- Directly estimates leakage path through attention.
- Useful for deciding which tokens should be treated as contaminated.

Cons:

- Attention is not perfect causal attribution.
- Requires access to attention probabilities.
- Can increase memory use if attention maps are retained.

### 5.4 Dual-Pass Feature Difference

Runs a no-LoRA base pass and regional-LoRA pass, then compares features.

```python
delta = x_regional - x_base
score = norm(delta, dim=-1)
```

Pros:

- Most direct measurement of actual divergence from the no-LoRA baseline.
- Captures direct and indirect LoRA effects.

Cons:

- Most expensive.
- Requires dual forward paths or checkpointed comparison points.
- May reduce speed significantly.

Recommended as an optional high-accuracy mode.

### 5.5 Final Prediction Delta

Compares the final model prediction from base and regional-LoRA passes.

```python
delta = pred_regional - pred_base
score = norm(delta, dim=-1)
```

Pros:

- Measures final denoising-update leakage.
- Useful for sampler-level correction.

Cons:

- Too late to influence earlier transformer attention within the same forward pass.
- Requires dual full-model prediction.

Recommended for optional diagnostics and sampler-level blending.

## 6. Normalization and Averaging

Each selected measurement source produces a per-token raw score:

```python
score_source[token]
```

Each source should be normalized independently before averaging.

Supported normalization modes:

### 6.1 Relative Norm

```python
relative_score = norm(delta, dim=-1) / (norm(reference, dim=-1) + eps)
```

Good default for comparing across layers with different activation magnitudes.

### 6.2 Per-Layer Min-Max

```python
score_norm = (score - score.min()) / (score.max() - score.min() + eps)
```

Useful for visualization and broad user tuning.

### 6.3 Percentile Normalization

```python
score_norm = score / (percentile(score, p) + eps)
score_norm = clamp(score_norm, 0, 1)
```

Recommended for robust thresholding.

Suggested default:

```text
normalization = relative_norm
percentile_clip = optional 95th percentile
```

If multiple sources are enabled:

```python
final_score = mean([normalized_source_1, normalized_source_2, ...])
```

Optional weighted average:

```python
final_score = sum(weight_i * normalized_source_i) / sum(weight_i)
```

## 7. Modified-Token Flagging

For each LoRA, maintain a per-token score and flag.

```python
new_modified = final_score > user_threshold
```

Only image tokens should be flagged for regional LoRA containment by default.

```python
modified_flags[lora_id][image_tokens] |= new_modified[image_tokens]
```

## 8. Flag Retention Modes

User-selectable retention behavior:

### 8.1 Sticky Flags

Once a token is flagged, it remains flagged for the rest of the generation.

```python
flags = flags | new_modified
```

Pros:

- Strongest leakage prevention.
- Simple.

Cons:

- False positives remain isolated.
- Can over-isolate tokens and increase seam risk.

Recommended for maximum containment.

### 8.2 Decaying Influence

Maintain a continuous influence score.

```python
influence = max(influence * decay, final_score)
flags = influence > threshold
```

Pros:

- Reduces permanent false positives.
- More flexible.

Cons:

- Slightly more complex.
- May allow late-stage leakage if decay is too strong.

Suggested controls:

```text
decay = 0.90 to 0.99
threshold = user-selected
```

### 8.3 Per-Layer Instantaneous Flags

Use only the current layer's measured score.

```python
flags = final_score > threshold
```

Pros:

- Minimal over-isolation.
- Useful for testing.

Cons:

- Weakest containment.

Recommended only as a diagnostic mode.

## 9. Asymmetric Attention Masking

The primary containment rule:

```text
Unmodified image tokens cannot attend to LoRA-modified image tokens.
```

In attention matrix terms:

```text
attention[query, key]
```

Block or penalize:

```text
query = unmodified image token
key   = modified image token
```

Do not block by default:

```text
modified image token → unmodified image token
```

This allows the LoRA region to use global scene context while preventing the rest of the image from reading LoRA-specific features.

## 10. Attention Bias Implementation

Before softmax:

```python
scores = q @ k.transpose(-2, -1) * scale
scores = scores + regional_attention_bias
attn = softmax(scores, dim=-1)
```

For each LoRA:

```python
modified = modified_flags[lora_id]
unmodified = ~modified
```

Build bias:

```python
bias[unmodified_queries, modified_keys] += -attention_isolation_strength
```

Suggested user control:

```text
attention_isolation_strength:
0      = off
2      = mild
5      = strong
20     = near-hard
10000  = hard block approximation
```

Default recommendation:

```text
5.0
```

## 11. Multiple Regions and Multiple LoRAs

Track modified flags separately per LoRA:

```python
modified_flags[lora_id] -> [seq_len]
modified_scores[lora_id] -> [seq_len]
```

This avoids conflating different regional concepts.

Example token state:

```text
0 = unmodified
1 = modified by LoRA A
2 = modified by LoRA B
3 = modified by both / conflict
```

Default rules:

```text
unmodified → modified_A: blocked or penalized
unmodified → modified_B: blocked or penalized
modified_A → unmodified: allowed
modified_B → unmodified: allowed
modified_A → modified_A: allowed
modified_B → modified_B: allowed
modified_A → modified_B: user-selectable
modified_B → modified_A: user-selectable
```

## 12. Cross-LoRA Attention Controls

For multiple character LoRAs, users should be able to choose cross-region behavior.

### 12.1 Allow Cross-LoRA Attention

```text
modified_A ↔ modified_B allowed
```

Pros:

- Better whole-scene coherence.
- Characters can interact more naturally.

Cons:

- Higher risk of identity bleed.

### 12.2 Penalize Cross-LoRA Attention

```text
modified_A → modified_B penalized
modified_B → modified_A penalized
```

Pros:

- Reduced identity mixing.
- Better for separate character identity preservation.

Cons:

- May reduce interaction coherence.

### 12.3 Hard Block Cross-LoRA Attention

```text
modified_A ─X→ modified_B
modified_B ─X→ modified_A
```

Pros:

- Maximum identity separation.

Cons:

- High risk of seams, poor interaction, broken contact.

Recommended default:

```text
soft penalty, strength 2-5
```

## 13. Text Token Handling

Default behavior:

```text
All image tokens may attend to text tokens.
Text tokens may attend according to the model's normal behavior.
```

Do not mark text tokens as LoRA-modified by default unless the LoRA explicitly targets the text encoder or text-token projections.

If text encoder LoRA support is added, expose separate controls:

```text
apply_text_encoder_lora: true/false
text_lora_is_global: true/false
text_lora_region_binding: experimental
```

Recommended default:

```text
Do not apply character LoRA to text encoder for regional isolation.
```

## 14. Box-to-Token Conversion

Given:

```text
pixel box: x0, y0, x1, y1
image size: W_px, H_px
token grid: W_tok, H_tok
```

Convert:

```python
tx0 = floor(x0 / W_px * W_tok)
tx1 = ceil(x1 / W_px * W_tok)
ty0 = floor(y0 / H_px * H_tok)
ty1 = ceil(y1 / H_px * H_tok)
```

Create row-major image-token mask:

```python
mask = torch.zeros(H_tok, W_tok, dtype=torch.bool)
mask[ty0:ty1, tx0:tx1] = True
mask = mask.flatten()
```

For Krea2-like f8 VAE compression and patch size 2:

```text
token grid ≈ image pixels / 16
```

Example:

```text
1024×1024 image → 64×64 image-token grid
```

## 15. Optional Sampler-Level Containment

For stricter isolation, optionally run base and regional predictions and blend prediction deltas spatially:

```python
pred_final = pred_base + latent_mask * (pred_regional - pred_base)
```

For maximum containment, maintain separate trajectories:

```text
base_latent_t: never sees LoRA
regional_latent_t: sees regional LoRA
combined output: soft-mask blend of both
```

This is more expensive but reduces contamination through shared latent state.

## 16. User Controls

Minimum controls:

```text
region bbox or mask
LoRA file
LoRA strength
delta measurement source(s)
delta threshold
normalization mode
flag retention mode
attention isolation strength
cross-LoRA attention behavior
```

Recommended advanced controls:

```text
per-source measurement weights
sticky vs decaying flags
decay rate
soft/hard attention blocking
include/exclude text tokens
include/exclude MLP deltas
include/exclude attention-propagated influence
debug heatmap output
sampler-level base/regional blending
dual-trajectory mode
```

## 17. Debug Outputs

The implementation should optionally output:

```text
per-LoRA modified-token heatmap
per-source delta heatmap
averaged modification score heatmap
attention-block heatmap
cross-LoRA conflict heatmap
final leakage estimate heatmap
```

For image display, token heatmaps should be upsampled to pixel space.

## 18. Recommended Defaults

```text
LoRA application:
  regional only

Delta measurement:
  per-LoRA-layer direct delta

Normalization:
  relative norm

Threshold:
  user-controlled; default around 0.05 to 0.15 for relative norm

Flag retention:
  sticky

Attention rule:
  unmodified image queries cannot attend to modified image keys

Attention isolation strength:
  5.0 soft penalty

Cross-LoRA attention:
  soft penalty, strength 2.0 to 5.0

Text tokens:
  globally available; not flagged

Sampler blending:
  off by default; optional high-containment mode
```

## 19. Expected Benefits

Compared to simple bbox masking:

```text
Better background continuity inside and around the region.
Less identity/style leakage outside the region.
Less seam risk than isolating all tokens inside a box.
More precise handling of multiple character LoRAs.
User-tunable tradeoff between coherence and containment.
```

## 20. Expected Failure Modes

```text
False-positive modified flags may over-isolate background tokens.
False-negative modified flags may allow leakage.
Hard attention blocking can create seams.
Text conditioning can still globally imply related concepts.
Shared latent trajectory can still carry regional influence into later steps.
Multiple LoRAs may contaminate each other without cross-LoRA controls.
Dual-pass and attention-tracking modes may be expensive.
```

## 21. Implementation Priority

Recommended development order:

```text
1. Regional LoRA delta masking by image-token region.
2. Direct per-LoRA-layer delta measurement.
3. Sticky modified-token flags.
4. Asymmetric attention penalty.
5. Debug heatmap output.
6. Multiple LoRA/region support.
7. Additional measurement sources.
8. Normalized multi-source averaging.
9. Cross-LoRA attention controls.
10. Optional sampler-level base/regional blending.
11. Optional dual-pass and dual-trajectory high-containment modes.
```

## 22. Core Design Summary

The key rule is:

```text
Do not isolate the whole box.
Isolate only tokens materially modified by the regional LoRA.
```

The key attention constraint is:

```text
Unmodified image tokens should not attend to LoRA-modified image tokens.
```

The key measurement rule is:

```text
If multiple delta measurement sources are enabled, normalize each source independently and average them before thresholding.
```

This should provide better regional identity containment while preserving background integration and image coherence.
