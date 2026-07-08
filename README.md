# ComfyUI Krea 2 Regional Attention-LoRA

Custom node pack for binding up to three character attention LoRAs to KJNodes bbox regions for Krea 2 workflows.

The implemented core uses regional prediction-delta blending:

1. Keep a base no-LoRA trajectory.
2. Keep a regional trajectory.
3. For each active region, compute a complete guided branch prediction, subtract the base guided prediction, and apply only the masked delta.
4. Resolve overlaps.
5. Pin everything outside the union mask back to the base trajectory after every step.

## Nodes

- `K2 BBox To Regional Mask`
- `K2 Regional Character LoRA`
- `K2 Regional LoRA Stack 3`
- `K2 Regional Layer LoRA Apply`
- `K2 Regional Attention LoRA Sampler`
- `K2 Regional Decode Composite`

## BBox Format

KJNodes bboxes default to `xywh`: `(x_min, y_min, width, height)`. The mask node also supports `xyxy`.

KJ displays boxes as `1`, `2`, `3`, etc. The mask node follows that numbering for `bbox_index`; `0` is also accepted as an alias for the first box for older workflows.

Krea 2 token snapping defaults to 16 output pixels, matching VAE compression 8 and DiT patch size 2.

## Sampling Modes

The sampler has two regional paths:

1. Strict adapter path. If the model exposes `k2_regional_velocity_predictor`, the sampler uses post-CFG regional prediction deltas and pins outside-region latent values back to the base trajectory after every denoising step. This is the strongest anti-leakage path.

2. Layer-injection fallback. If no strict adapter is present and `execution_mode=auto`, the sampler clones the model, installs temporary layer hooks, applies LoRA activation deltas only on masked regional token streams, runs normal Comfy sampling, then pins the final latent outside the union mask back to `base_samples`. This makes the node usable in ordinary ComfyUI Krea workflows while avoiding a global character LoRA load.

An adapter object can be attached to the model as `k2_regional_velocity_predictor` and must provide:

```python
schedule(model, steps, seed) -> Sequence[float]
guided_predict(model, x, positive, negative, cfg, sigma_or_t) -> torch.Tensor
```

## Layer-Injection Fallback

The fallback is based on the "stamp only where the token writes back" idea:

- Q/K/V-only masking is not used for isolation.
- The default `attn_out_mlp` target masks LoRA deltas on attention output projections and MLP layers.
- `outside_strength=0.0` keeps image-token LoRA deltas out of unassigned regions.
- `text_token_strength=0.0` avoids applying character LoRA deltas to text tokens in the fallback.
- Hooks are installed on a cloned model and removed after each diffusion-model call by ComfyUI's wrapper mechanism.

This fallback is not as mathematically strict as per-step sampler pinning because attention can still mix information after a regional write. It mitigates that weakness by reapplying the regional write repeatedly across layers/steps and by final-pinning the latent outside the region to the base result.

You can also use `K2 Regional Layer LoRA Apply` directly before a normal sampler. The all-in-one sampler node is usually easier because it also returns `base_samples`, `union_mask`, and applies final latent pinning.

## Testing

Run:

```bash
/home/wolfhard/ComfyUI/311venv/bin/python run_tests.py
```

The tests cover KJ bbox conversion, xyxy conversion, empty/disabled regions, fake global LoRA deltas, exact outside pinning, three disjoint regions, overlap modes, LoRA key filtering, writeback key selection, and batch shapes.
