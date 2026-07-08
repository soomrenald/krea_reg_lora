from __future__ import annotations

import torch

from krea_region_lora.layer_injection import sequence_mask_for_region
from krea_region_lora.masks import coerce_bbox_list, region_from_bbox
from krea_region_lora.tracking import RegionalRuntimeState
from krea_region_lora.types import KreaRegionalLora, KreaRegionalLoraStack


def make_region(bbox=(16, 16, 16, 16), *, width=64, height=64):
    return region_from_bbox([bbox], width=width, height=height, feather_px=0, snap_to_krea_token_grid=False)


def make_lora(region, name="char", **kwargs):
    return KreaRegionalLora(
        region=region,
        positive=[],
        negative=[],
        lora_name=name,
        threshold=kwargs.pop("threshold", 0.01),
        **kwargs,
    )


def test_kj_ideogram_nested_bboxes_are_pixel_xywh():
    data = [[{"x": 10, "y": 20, "width": 30, "height": 40}, {"x": 50, "y": 60, "width": 70, "height": 80}]]
    assert coerce_bbox_list(data) == [(10.0, 20.0, 30.0, 40.0), (50.0, 60.0, 70.0, 80.0)]


def test_bbox_index_is_one_based_with_zero_alias():
    data = [[{"x": 0, "y": 0, "width": 16, "height": 16}, {"x": 32, "y": 32, "width": 16, "height": 16}]]
    zero = region_from_bbox(data, width=64, height=64, bbox_index=0, feather_px=0, snap_to_krea_token_grid=False)
    one = region_from_bbox(data, width=64, height=64, bbox_index=1, feather_px=0, snap_to_krea_token_grid=False)
    two = region_from_bbox(data, width=64, height=64, bbox_index=2, feather_px=0, snap_to_krea_token_grid=False)
    assert zero.pixel_bbox == (0, 0, 16, 16)
    assert one.pixel_bbox == (0, 0, 16, 16)
    assert two.pixel_bbox == (32, 32, 48, 48)


def test_token_and_latent_masks_match_krea_grid_defaults():
    region = make_region()
    assert region.pixel_mask.shape == (1, 64, 64)
    assert region.latent_mask.shape == (1, 1, 8, 8)
    assert region.token_mask.shape == (1, 16, 1)
    assert torch.all(region.latent_mask[:, :, 2:4, 2:4] == 1)


def test_sequence_mask_places_image_tokens_at_sequence_tail():
    region = make_region()
    regional = make_lora(region)
    x = torch.zeros((1, 20, 8))
    mask = sequence_mask_for_region(regional, x, outside_strength=0.0, text_token_strength=0.0)
    assert mask.shape == (1, 20, 1)
    assert torch.count_nonzero(mask[:, :4]) == 0
    assert torch.count_nonzero(mask[:, 4:]) == torch.count_nonzero(region.token_mask)


def test_direct_delta_tracking_marks_only_changed_region_tokens():
    region = make_region()
    regional = make_lora(region, threshold=0.01)
    stack = KreaRegionalLoraStack((regional,), attention_isolation_strength=5.0)
    state = RegionalRuntimeState(stack)
    delta = torch.zeros((1, 20, 4))
    reference = torch.ones((1, 20, 4))
    mask = sequence_mask_for_region(regional, delta, outside_strength=0.0, text_token_strength=0.0)
    delta = delta + mask * 1.0
    flags = state.update_from_delta(regional, delta, reference)
    assert flags.shape == (1, 16)
    assert int(flags.sum()) == int(torch.count_nonzero(region.token_mask))


def test_attention_bias_blocks_unmodified_queries_to_modified_keys():
    region = make_region()
    regional = make_lora(region, threshold=0.01)
    stack = KreaRegionalLoraStack((regional,), attention_isolation_strength=7.0)
    state = RegionalRuntimeState(stack)
    state.flags[id(regional)] = region.token_mask.squeeze(-1).bool()
    bias = state.build_attention_bias(batch=1, q_len=20, k_len=20, heads=2, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is not None
    assert bias.shape == (2, 20, 20)
    image_start = 4
    modified = state.flags[id(regional)][0]
    modified_index = image_start + int(torch.nonzero(modified)[0])
    unmodified_index = image_start + int(torch.nonzero(~modified)[0])
    assert bias[0, unmodified_index, modified_index] == -7.0
    assert bias[0, modified_index, unmodified_index] == 0.0


def test_cross_lora_penalty_is_separate_from_background_penalty():
    r1 = make_region((0, 0, 16, 16))
    r2 = make_region((32, 32, 16, 16))
    l1 = make_lora(r1, "one")
    l2 = make_lora(r2, "two")
    state = RegionalRuntimeState(KreaRegionalLoraStack((l1, l2), attention_isolation_strength=0.0, cross_lora_mode="penalize", cross_lora_strength=3.0))
    state.flags[id(l1)] = r1.token_mask.squeeze(-1).bool()
    state.flags[id(l2)] = r2.token_mask.squeeze(-1).bool()
    bias = state.build_attention_bias(batch=1, q_len=16, k_len=16, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is not None
    i = int(torch.nonzero(state.flags[id(l1)][0])[0])
    j = int(torch.nonzero(state.flags[id(l2)][0])[0])
    assert bias[0, i, j] == -3.0
    assert bias[0, j, i] == -3.0
