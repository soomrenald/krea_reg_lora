from __future__ import annotations

import torch

from torch import nn

from krea_region_lora.conditioning import PromptTokenRange, RegionalPromptRuntimeState, build_conditioning_stack, conditioning_debug_preview, region_ids_match_lora
from krea_region_lora.layer_injection import _eligible_linears, sequence_mask_for_region
from krea_region_lora.masks import coerce_bbox_list, region_from_bbox
from krea_region_lora.tracking import RegionalRuntimeState
from krea_region_lora.types import KreaRegionalConditioning, KreaRegionalLora, KreaRegionalLoraStack, parse_measurement_sources


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


def test_sequence_mask_places_image_tokens_after_text_before_padding():
    region = make_region()
    regional = make_lora(region)
    x = torch.zeros((1, 22, 8))
    mask = sequence_mask_for_region(regional, x, outside_strength=0.0, text_token_strength=0.0, text_len=3, img_len=16)
    assert mask.shape == (1, 22, 1)
    assert torch.count_nonzero(mask[:, :3]) == 0
    assert torch.count_nonzero(mask[:, 3:19]) == torch.count_nonzero(region.token_mask)
    assert torch.count_nonzero(mask[:, 19:]) == 0


def test_direct_delta_tracking_marks_only_changed_region_tokens():
    region = make_region()
    regional = make_lora(region, threshold=0.01)
    stack = KreaRegionalLoraStack((regional,), attention_isolation_strength=5.0)
    state = RegionalRuntimeState(stack)
    state.text_len = 3
    state.img_len = 16
    delta = torch.zeros((1, 22, 4))
    reference = torch.ones((1, 22, 4))
    mask = sequence_mask_for_region(regional, delta, outside_strength=0.0, text_token_strength=0.0, text_len=3, img_len=16)
    delta = delta + mask * 1.0
    flags = state.update_from_delta(regional, delta, reference)
    assert flags.shape == (1, 16)
    assert int(flags.sum()) == int(torch.count_nonzero(region.token_mask))


def test_parse_measurement_sources_accepts_hidden_state_delta_aliases():
    assert parse_measurement_sources("direct_delta, hidden_delta") == ("direct_delta", "hidden_state_delta")
    assert parse_measurement_sources(["hidden_state"]) == ("hidden_state_delta",)


def test_parse_measurement_sources_accepts_final_prediction_delta_as_diagnostic():
    assert parse_measurement_sources("final_prediction_delta") == ("final_prediction_delta",)


def test_multi_source_tracking_averages_normalized_measurements():
    region = make_region()
    regional = make_lora(region, threshold=0.4)
    state = RegionalRuntimeState(KreaRegionalLoraStack((regional,), attention_isolation_strength=5.0))
    state.text_len = 3
    state.img_len = 16
    direct = torch.zeros((1, 22, 4))
    hidden = torch.zeros((1, 22, 4))
    reference = torch.ones((1, 22, 4))
    mask = sequence_mask_for_region(regional, hidden, outside_strength=0.0, text_token_strength=0.0, text_len=3, img_len=16)
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
    state.text_len = 3
    state.img_len = 16
    state.flags[id(regional)] = region.token_mask.squeeze(-1).bool()
    bias = state.build_attention_bias(batch=1, q_len=22, k_len=22, heads=2, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is not None
    assert bias.shape == (2, 22, 22)
    image_start = 3
    modified = state.flags[id(regional)][0]
    modified_index = image_start + int(torch.nonzero(modified)[0])
    unmodified_index = image_start + int(torch.nonzero(~modified)[0])
    assert bias[0, unmodified_index, modified_index] == -7.0
    assert bias[0, modified_index, unmodified_index] == 0.0
    assert bias[0, 20, modified_index] == 0.0


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
    assert region.img_len == 16
    assert region.text_len is None


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


def test_runtime_layout_report_includes_padding_span():
    region = make_region()
    regional = make_lora(region)
    state = RegionalRuntimeState(KreaRegionalLoraStack((regional,), attention_isolation_strength=5.0))
    state.text_len = 3
    state.img_len = 16
    layout = state.layout_for(regional, 22)
    assert layout is not None
    assert (layout.img_start, layout.img_end, layout.pad_len) == (3, 19, 3)
    report = state.report()
    assert "seq_len=22 text_len=3 img_len=16 pad_len=3 img_start=3 img_end=19" in report



def make_prompt_conditioning(region, *, tokens=2, mode="forbid", strength=5.0, outside_strength=0.0):
    return KreaRegionalConditioning(
        region=region,
        conditioning=[[torch.zeros((1, tokens, 4)), {}]],
        text="regional subject",
        region_id=region.region_id,
        strength=1.0,
        outside_strength=outside_strength,
        prompt_attention_mode=mode,
        prompt_attention_strength=strength,
    )


def make_prompt_state(region, *, mode="forbid", outside_strength=0.0):
    prompt = make_prompt_conditioning(region, mode=mode, outside_strength=outside_strength)
    stack = build_conditioning_stack([[torch.zeros((1, 3, 4)), {}]], [prompt])
    state = RegionalPromptRuntimeState(stack)
    state.global_text_len = 3
    state.text_len = 5
    state.img_len = 16
    state.image_rows = 4
    state.image_cols = 4
    state.prompt_ranges = [PromptTokenRange(region.region_id, 3, 5, prompt)]
    return state, prompt


def test_prompt_attention_bias_penalizes_non_region_image_queries_to_prompt_keys():
    region = make_region((0, 0, 32, 64))
    state, _prompt = make_prompt_state(region, mode="penalize")
    bias = state.build_prompt_attention_bias(batch=1, q_len=24, k_len=24, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is not None
    image_start = 5
    inside_image_index = image_start + int(torch.nonzero(region.token_mask[0, :, 0] > 0)[0])
    outside_image_index = image_start + int(torch.nonzero(region.token_mask[0, :, 0] == 0)[0])
    prompt_key = 3
    assert bias[0, inside_image_index, prompt_key] == 0.0
    assert bias[0, outside_image_index, prompt_key] == -5.0
    assert bias[0, 23, prompt_key] == 0.0


def test_prompt_attention_bias_forbid_uses_large_negative_bias():
    region = make_region((0, 0, 32, 64))
    state, _prompt = make_prompt_state(region, mode="forbid")
    bias = state.build_prompt_attention_bias(batch=1, q_len=24, k_len=24, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    outside_image_index = 5 + int(torch.nonzero(region.token_mask[0, :, 0] == 0)[0])
    assert bias is not None
    assert bias[0, outside_image_index, 3] <= -10000.0


def test_prompt_attention_bias_does_not_leak_unrelated_region_prompt_tokens():
    r1 = make_region((0, 0, 16, 16))
    r2 = make_region((48, 48, 16, 16))
    p1 = make_prompt_conditioning(r1, tokens=2, mode="forbid")
    p2 = make_prompt_conditioning(r2, tokens=2, mode="forbid")
    state = RegionalPromptRuntimeState(build_conditioning_stack([], [p1, p2]))
    state.text_len = 7
    state.img_len = 16
    state.image_rows = 4
    state.image_cols = 4
    state.prompt_ranges = [PromptTokenRange(r1.region_id, 3, 5, p1), PromptTokenRange(r2.region_id, 5, 7, p2)]
    bias = state.build_prompt_attention_bias(batch=1, q_len=23, k_len=23, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is not None
    r1_image = 7 + int(torch.nonzero(r1.token_mask[0, :, 0] > 0)[0])
    r2_prompt_key = 5
    assert bias[0, r1_image, r2_prompt_key] < 0.0
    assert bias[0, r1_image, 3] == 0.0


def test_prompt_attention_bias_all_zero_mask_produces_no_bias():
    region = make_region(None)
    state, _prompt = make_prompt_state(region, mode="forbid")
    bias = state.build_prompt_attention_bias(batch=1, q_len=24, k_len=24, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is None


def test_prompt_attention_bias_all_one_mask_has_no_text_or_padding_bias():
    region = make_region((0, 0, 64, 64))
    state, _prompt = make_prompt_state(region, mode="forbid")
    bias = state.build_prompt_attention_bias(batch=1, q_len=24, k_len=24, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is None


def test_prompt_attention_bias_left_half_mask_only_penalizes_right_half_image_queries():
    region = make_region((0, 0, 32, 64))
    state, _prompt = make_prompt_state(region, mode="penalize")
    bias = state.build_prompt_attention_bias(batch=1, q_len=24, k_len=24, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is not None
    image_start = 5
    left_positions = torch.nonzero(region.token_mask[0, :, 0] > 0).flatten()
    right_positions = torch.nonzero(region.token_mask[0, :, 0] == 0).flatten()
    assert torch.all(bias[0, image_start + left_positions, 3] == 0.0)
    assert torch.all(bias[0, image_start + right_positions, 3] == -5.0)


def test_same_region_prompt_and_lora_masks_align_in_debug_report():
    region = make_region((0, 0, 32, 64))
    prompt = make_prompt_conditioning(region)
    conditioning_stack = build_conditioning_stack([], [prompt])
    lora_stack = KreaRegionalLoraStack((make_lora(region, "same"),))
    report = region_ids_match_lora(conditioning_stack, lora_stack)
    assert "prompt_lora_mask_align=True" in report
    assert f"prompt_region_id={region.region_id}" in report
    assert f"lora_region_id={region.region_id}" in report


def test_all_zero_region_mask_produces_no_lora_effect():
    region = make_region(None)
    regional = make_lora(region, threshold=0.01)
    x = torch.zeros((1, 16, 4))
    mask = sequence_mask_for_region(regional, x, outside_strength=0.0, text_token_strength=0.0, text_len=0, img_len=16)
    assert mask is not None
    assert int(torch.count_nonzero(mask)) == 0


def test_all_one_lora_mask_affects_all_image_tokens_and_no_text_or_padding():
    region = make_region((0, 0, 64, 64))
    regional = make_lora(region)
    x = torch.zeros((1, 22, 4))
    mask = sequence_mask_for_region(regional, x, outside_strength=0.0, text_token_strength=0.0, text_len=3, img_len=16)
    assert mask is not None
    assert int(torch.count_nonzero(mask[:, :3])) == 0
    assert int(torch.count_nonzero(mask[:, 3:19])) == 16
    assert int(torch.count_nonzero(mask[:, 19:])) == 0


def test_left_half_lora_mask_affects_only_left_half_image_tokens():
    region = make_region((0, 0, 32, 64))
    regional = make_lora(region)
    x = torch.zeros((1, 19, 4))
    mask = sequence_mask_for_region(regional, x, outside_strength=0.0, text_token_strength=0.0, text_len=3, img_len=16)
    assert mask is not None
    image_mask = mask[:, 3:19, 0]
    assert int(torch.count_nonzero(image_mask)) == 8
    assert torch.equal(image_mask.bool(), region.token_mask.squeeze(-1).bool())


def test_lora_attention_isolation_forbid_creates_large_negative_bias():
    region = make_region()
    regional = make_lora(region, threshold=0.01)
    stack = KreaRegionalLoraStack((regional,), attention_isolation_mode="forbid", attention_isolation_strength=7.0)
    state = RegionalRuntimeState(stack)
    state.text_len = 3
    state.img_len = 16
    state.flags[id(regional)] = region.token_mask.squeeze(-1).bool()
    bias = state.build_attention_bias(batch=1, q_len=22, k_len=22, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    modified = state.flags[id(regional)][0]
    modified_index = 3 + int(torch.nonzero(modified)[0])
    unmodified_index = 3 + int(torch.nonzero(~modified)[0])
    assert bias is not None
    assert bias[0, unmodified_index, modified_index] <= -10000.0


def test_modified_outward_mode_controls_modified_queries_attending_outward_separately():
    region = make_region()
    regional = make_lora(region, threshold=0.01)
    stack = KreaRegionalLoraStack(
        (regional,),
        attention_isolation_mode="none",
        modified_outward_mode="forbid",
        modified_outward_strength=7.0,
    )
    state = RegionalRuntimeState(stack)
    state.text_len = 3
    state.img_len = 16
    state.flags[id(regional)] = region.token_mask.squeeze(-1).bool()
    bias = state.build_attention_bias(batch=1, q_len=22, k_len=22, heads=1, device=torch.device("cpu"), dtype=torch.float32)
    modified = state.flags[id(regional)][0]
    modified_index = 3 + int(torch.nonzero(modified)[0])
    unmodified_index = 3 + int(torch.nonzero(~modified)[0])
    assert bias is not None
    assert bias[0, modified_index, unmodified_index] <= -10000.0
    assert bias[0, unmodified_index, modified_index] == 0.0
