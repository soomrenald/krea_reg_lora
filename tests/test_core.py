from __future__ import annotations

import torch

from torch import nn

from krea_region_lora.conditioning import build_conditioning_stack, conditioning_debug_preview, region_ids_match_lora
from krea_region_lora.layer_injection import _eligible_linears, sequence_mask_for_region
from krea_region_lora.masks import coerce_bbox_list, region_from_bbox
from krea_region_lora.tracking import RegionalRuntimeState
from krea_region_lora.types import KreaRegionalLora, KreaRegionalLoraStack, parse_measurement_sources


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


def test_parse_measurement_sources_accepts_hidden_state_delta_aliases():
    assert parse_measurement_sources("direct_delta, hidden_delta") == ("direct_delta", "hidden_state_delta")
    assert parse_measurement_sources(["hidden_state"]) == ("hidden_state_delta",)


def test_multi_source_tracking_averages_normalized_measurements():
    region = make_region()
    regional = make_lora(region, threshold=0.4)
    state = RegionalRuntimeState(KreaRegionalLoraStack((regional,), attention_isolation_strength=5.0))
    direct = torch.zeros((1, 20, 4))
    hidden = torch.zeros((1, 20, 4))
    reference = torch.ones((1, 20, 4))
    mask = sequence_mask_for_region(regional, hidden, outside_strength=0.0, text_token_strength=0.0)
    hidden = hidden + mask * 1.0
    flags = state.update_from_measurements(regional, [(direct, reference), (hidden, reference)])
    assert flags.shape == (1, 16)
    assert int(flags.sum()) == int(torch.count_nonzero(region.token_mask))


def test_eligible_linears_skip_krea_text_conditioning_modules():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.txtfusion = nn.Sequential(nn.Linear(4, 4))
            self.txtmlp = nn.Sequential(nn.Linear(4, 4))
            self.tmlp = nn.Sequential(nn.Linear(4, 4))
            self.tproj = nn.Sequential(nn.Linear(4, 4))
            self.blocks = nn.ModuleList([nn.Module()])
            self.blocks[0].attn = nn.Module()
            self.blocks[0].attn.wo = nn.Linear(4, 4)
            self.blocks[0].mlp = nn.Module()
            self.blocks[0].mlp.down = nn.Linear(4, 4)

    eligible = _eligible_linears(Dummy(), "attn_out_mlp")
    assert "blocks.0.attn.wo" in eligible
    assert "blocks.0.mlp.down" in eligible
    assert not any(name.startswith(("txtfusion.", "txtmlp.", "tmlp.", "tproj.")) for name in eligible)


def test_sequence_mask_ignores_sequence_shorter_than_image_tokens():
    region = make_region()
    regional = make_lora(region)
    x = torch.zeros((1, 12, 8))
    assert sequence_mask_for_region(regional, x, outside_strength=0.0, text_token_strength=0.0) is None


def test_direct_delta_tracking_ignores_sequence_shorter_than_image_tokens():
    region = make_region()
    regional = make_lora(region, threshold=0.01)
    state = RegionalRuntimeState(KreaRegionalLoraStack((regional,), attention_isolation_strength=5.0))
    delta = torch.ones((1, 12, 4))
    reference = torch.ones((1, 12, 4))
    flags = state.update_from_delta(regional, delta, reference)
    assert flags.shape == (1, 16)
    assert int(flags.sum()) == 0
    assert state.flags == {}


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



def test_region_carries_canonical_identity_and_normalized_bbox():
    region = make_region((16, 16, 16, 16), width=64, height=64)
    assert region.region_id == "bbox_0_16_16_32_32"
    assert region.pixel_bbox == (16, 16, 32, 32)
    assert region.normalized_bbox == (0.25, 0.25, 0.5, 0.5)
    assert region.feather_px == 0


def test_regional_conditioning_stack_preserves_order_and_region_ids():
    r1 = make_region((0, 0, 16, 16))
    r2 = make_region((32, 32, 16, 16))
    c1 = type("Regional", (), {"region": r1, "conditioning": [], "text": "woman", "strength": 1.0, "outside_strength": 0.0, "feather_px": None})()
    c2 = type("Regional", (), {"region": r2, "conditioning": [], "text": "man", "strength": 1.0, "outside_strength": 0.0, "feather_px": None})()
    stack = build_conditioning_stack([[torch.zeros((1, 1, 4)), {}]], [c1, c2])
    assert [r.region.region_id for r in stack.regions] == [r1.region_id, r2.region_id]


def test_conditioning_debug_preview_uses_region_masks():
    r1 = make_region((0, 0, 16, 16))
    c1 = type("Regional", (), {"region": r1, "conditioning": [], "text": "woman", "strength": 1.0, "outside_strength": 0.0, "feather_px": None})()
    stack = build_conditioning_stack([], [c1])
    image, report = conditioning_debug_preview(stack)
    assert image.shape == (1, 64, 64, 3)
    assert float(image.max()) == 1.0
    assert r1.region_id in report


def test_prompt_lora_match_report_uses_shared_region_id():
    region = make_region()
    c1 = type("Regional", (), {"region": region, "conditioning": [], "text": "woman", "strength": 1.0, "outside_strength": 0.0, "feather_px": None})()
    conditioning_stack = build_conditioning_stack([], [c1])
    lora_stack = KreaRegionalLoraStack((make_lora(region, "same"),))
    report = region_ids_match_lora(conditioning_stack, lora_stack)
    assert f"region_id={region.region_id} shared_with_lora=True" in report
