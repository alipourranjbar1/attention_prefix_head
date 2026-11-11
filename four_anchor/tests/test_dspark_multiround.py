from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import random

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from torch.nn import functional as F

from four_anchor.bonus_addon_objective import first_anchors, redraft_anchors
from four_anchor.cached_loss import teacher_cache
from four_anchor.dspark_loss import projected_dspark_loss
from four_anchor.dspark_model import DSparkDraftConfig, FourAnchorDSpark
from four_anchor.dspark_train import DSparkTrainConfig, build_config
from four_anchor.model import FourAnchorDFlash
from four_anchor.packed_objective import PackedTask
from four_anchor.packed_train import metric_row
from four_anchor.tests.test_packed_training import batch_for, tiny_config


def setup(components):
    target, original, runner = components
    dc = DSparkDraftConfig("tiny", [0, 1], 63, num_draft_layers=2,
        num_proposals=15, anchor_layout="bonus_addon", format_version=4, markov_rank=4)
    draft = FourAnchorDSpark(dc, target)
    cfg = DSparkTrainConfig(**asdict(tiny_config(num_proposals=15, anchor_layout="bonus_addon")),
                           markov_rank=4)
    return target, draft, runner, cfg


def test_markov_chain_uses_previous_generated_token(components):
    _, draft, _, _ = setup(components)
    hidden = torch.randn(3, 15, draft.hidden_size)
    seeds = torch.tensor([4, 11, 23])
    actual = draft.greedy_proposals(hidden, seeds)
    expected, prev = [], seeds
    for pos in range(15):
        logits = draft.lm_head(hidden[:, pos]) + draft.markov_head.markov_w2(draft.markov_head.markov_w1(prev))
        prev = logits.argmax(-1)
        expected.append(prev)
    assert torch.equal(actual[0], torch.stack(expected, 1))
    assert all(x.shape == (3, 15) for x in actual)
    assert not any(x.requires_grad for x in actual)
    # Make the previous-token dependency unambiguous rather than relying on
    # random initial logits happening to select different tokens.
    with torch.no_grad():
        draft.markov_head.markov_w1.weight.zero_()
        draft.markov_head.markov_w1.weight[4, 0] = 100
        draft.markov_head.markov_w1.weight[11, 1] = 100
        draft.markov_head.markov_w2.weight.zero_()
        draft.markov_head.markov_w2.weight[17, 0] = 100
        draft.markov_head.markov_w2.weight[18, 1] = 100
    assert draft.greedy_proposals(hidden[:2], seeds[:2])[0][:, 0].tolist() == [17, 18]


@pytest.mark.parametrize("chunk", [1, 7, 100])
def test_full_vocabulary_loss_and_gradients_match_dense_eager(components, chunk):
    target, draft, _, _ = setup(components)
    other = deepcopy(draft)
    hidden = torch.randn(2, 5, draft.hidden_size, requires_grad=True)
    h2 = hidden.detach().clone().requires_grad_(True)
    prev = torch.randint(0, 64, (2, 5))
    probabilities = torch.softmax(torch.randn(11, 64), -1)
    targets = (probabilities, probabilities.argmax(-1))
    indices = torch.tensor([[1, 2, 4, 7, 8], [0, 2, 9, 4, 6]])
    weights = torch.tensor([[1., .8, 0., .4, .2], [.7, .6, .5, 0., 0.]])
    mix, alpha = {"ce": .1, "tv": .9, "kl_div": .2}, .6
    loss, stats, confidence_stats = projected_dspark_loss(draft, hidden, prev, targets,
        indices, weights, mix, alpha, chunk)
    logits, confidence = other.project(h2, prev)
    logq = logits.float().log_softmax(-1)
    p = probabilities[indices]
    ce = -logq.gather(-1, targets[1][indices][..., None]).squeeze(-1)
    overlap = torch.minimum(logq.exp(), p).sum(-1).clamp(0., 1.)
    kl = F.kl_div(logq, p, reduction="none").sum(-1)
    bce = F.binary_cross_entropy_with_logits(confidence.float(), overlap.detach(), reduction="none")
    expected = ((.1 * ce + .9 * (1 - overlap) + .2 * kl + alpha * bce) * weights).sum()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(stats, (torch.stack((ce, 1 - overlap, kl), -1) * weights[..., None]).sum((0, 1)))
    torch.testing.assert_close(confidence_stats[0], (bce * weights).sum())
    loss.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, h2.grad)
    for (name, param), (_, p2) in zip(draft.named_parameters(), other.named_parameters()):
        if param.grad is not None:
            torch.testing.assert_close(param.grad, p2.grad, msg=name)
    for module in (draft.markov_head.markov_w1, draft.markov_head.markov_w2, draft.confidence_head.proj):
        assert module.weight.grad.abs().sum() > 0
    assert not any(p.grad is not None for p in target.parameters())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_two_round_gradients_cached_reference_and_zero_restart(components, monkeypatch, checkpointing):
    target, draft, _, cfg = setup(components)
    batch = batch_for(components)
    draft.gradient_checkpointing = checkpointing
    other = deepcopy(draft)
    monkeypatch.setattr("four_anchor.dspark_objective.rejection_boundary", lambda x: torch.zeros(len(x), dtype=torch.long))
    values = []
    for model, mode in ((draft, "reference"), (other, "cached")):
        task = PackedTask(model, target.model.norm, replace(cfg, packed_execution=mode, boundary_exploration=0), random.Random(5))
        num, den, metrics = task(batch)
        assert metrics.shape == (17,)
        assert metrics[10] == 0 and metrics[5] > 0 and metrics[9] > 0
        assert task.last_redraft_blocks == cfg.max_anchors
        (num / den).backward()
        for name in ("draft_anchor_proj.weight", "markov_head.markov_w1.weight", "markov_head.markov_w2.weight", "confidence_head.proj.weight"):
            assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name
        row = metric_row(torch.cat((metrics, metrics.new_tensor([task.last_redraft_blocks]))))
        assert row["first_confidence_bce"] > 0 and row["second_confidence_bce"] > 0
        values.append((num, den, metrics))
    for x, y in zip(*values):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=3e-5)
    for (name, p), (_, q) in zip(draft.named_parameters(), other.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=5e-4, msg=name)


@pytest.mark.parametrize("boundary", range(15))
def test_all_restart_boundaries_keep_draft_only_anchors_and_isolate_future(components, boundary):
    _, draft, _, _ = setup(components)
    batch = batch_for(components)
    starts, docs = torch.tensor([5]), torch.tensor([0])
    a1, bonus = first_anchors(batch, starts, docs, draft.hidden_size)
    def hidden(b, anchors, s, **kw):
        memory = draft.packed_memory(b["features"][None], b["positions"])
        return draft.packed_hidden(b["features"][None], b["positions"], b["documents"],
            anchors, s, starts - 1, docs, memory=memory, **kw)
    first = hidden(batch, a1, starts, bonus_ids=bonus, return_all=True)
    tokens = draft.greedy_proposals(first[:, 5:], bonus[:, 0])[0]
    keep = torch.tensor([boundary])
    a2 = redraft_anchors(a1, bonus, starts, keep, first, tokens)
    assert not a2.target_features.any()
    assert not a2.draft_features.requires_grad
    assert (a2.source == 2).all()
    second = hidden(batch, a2, starts + keep)
    assert first.shape[1] == 20 and second.shape[1] == 15
    altered = {**batch, "features": batch["features"].clone()}
    altered["features"][4:] = torch.randn_like(altered["features"][4:]) * 100
    torch.testing.assert_close(first, hidden(altered, a1, starts, bonus_ids=bonus, return_all=True), atol=0, rtol=0)
    torch.testing.assert_close(second, hidden(altered, a2, starts + keep), atol=0, rtol=0)
    expected_seed = bonus[:, 0] if boundary == 0 else tokens[:, boundary - 1]
    assert torch.equal(a2.ids[:, -1], expected_seed)


def test_loss_uses_previous_tokens_and_only_replaced_suffix(components, monkeypatch):
    import four_anchor.dspark_objective as objective
    target, draft, _, cfg = setup(components)
    batch = batch_for(components)
    cfg = replace(cfg, max_anchors=1, boundary_exploration=0)
    seen, real = [], objective.projected_dspark_loss
    def capture(model, h, prev, targets, ti, w, *args):
        seen.append((prev.clone(), ti.clone(), w.clone()))
        return real(model, h, prev, targets, ti, w, *args)
    monkeypatch.setattr(objective, "projected_dspark_loss", capture)
    PackedTask(draft, target.model.norm, cfg, random.Random(2))(batch)
    assert len(seen) == 2
    for previous, teacher_index, weights in seen:
        assert previous.shape == (1, 15)
        assert torch.equal(previous, batch["ids"][teacher_index])
    assert seen[1][2].sum() <= seen[0][2].sum() + 1e-6


def test_checkpoint_identity_and_head_roundtrip(components, tmp_path):
    target, draft, _, _ = setup(components)
    draft.save(tmp_path)
    meta = json.loads((tmp_path / "four_anchor_config.json").read_text())
    assert meta["architecture"] == "FourAnchorDSpark"
    assert meta["draft_config"]["markov_rank"] == 4
    restored = FourAnchorDSpark.load(tmp_path, target, trainable_dtype=torch.float32)
    for name, value in draft.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="FourAnchorDFlash"):
        FourAnchorDFlash.load(tmp_path, target)
    assert not restored.lm_head.weight.requires_grad


def test_bfloat16_autocast_keeps_trainable_master_weights_and_heads(components, tmp_path):
    target, draft, _, cfg = setup(components)
    target.bfloat16()
    # The real trainer borrows BF16 target weights but retains FP32 masters for
    # every trainable parameter, including both newly attached DSpark heads.
    for p in draft.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    draft.gradient_checkpointing = True
    batch = batch_for(components)
    task = PackedTask(draft, target.model.norm, replace(cfg, packed_execution="cached"), random.Random(4))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, den, _ = task(batch)
    (loss / den).backward()
    assert torch.isfinite(loss)
    for name, p in draft.named_parameters():
        if p.requires_grad:
            assert p.dtype == torch.float32
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    draft.save(tmp_path)
    restored = FourAnchorDSpark.load(tmp_path, target, trainable_dtype=torch.float32)
    for (name, p), (_, q) in zip(draft.named_parameters(), restored.named_parameters()):
        assert torch.equal(p, q), name


def test_second_loss_stops_at_original_window_end(components, monkeypatch):
    import four_anchor.dspark_objective as objective
    from four_anchor.packed_data import TeacherRecords, collate_documents
    target, draft, runner, cfg = setup(components)
    cfg = replace(cfg, max_anchors=1, max_sequence_length=64, boundary_exploration=0)
    rows = Dataset.from_dict({"input_ids": [list(range(1, 41))], "loss_mask": [[1] * 40]})
    batch = collate_documents([TeacherRecords(rows, cfg, target.config, runner)[0]])
    class FixedStart(random.Random):
        def sample(self, population, k):
            return [5]
    # Choose a seven-token retained prefix that is valid for the cached teacher.
    monkeypatch.setattr(draft, "greedy_proposals", lambda h, seed, **kwargs:
        (batch["ids"][5:20][None], torch.zeros(1, 15), torch.zeros(1, 15)))
    monkeypatch.setattr(objective, "rejection_boundary", lambda x: torch.tensor([7]))
    seen, real = [], objective.projected_dspark_loss
    def capture(model, h, prev, targets, ti, w, *args):
        seen.append((prev.clone(), ti.clone(), w.clone()))
        return real(model, h, prev, targets, ti, w, *args)
    monkeypatch.setattr(objective, "projected_dspark_loss", capture)
    task = PackedTask(draft, target.model.norm, cfg, FixedStart())
    _, _, metrics = task(batch)
    assert metrics[10] == 7
    assert (seen[0][2] > 0).sum() == 15
    assert (seen[1][2] > 0).sum() == 8
    assert seen[0][1][0, 0] == 4  # predict position 5 using state/token at 4
    assert seen[1][1][0, 0] == 11  # predict position 12 using state/token at 11
    assert seen[1][1][0, 7] == 18  # last supervised replacement predicts position 19


def test_actual_training_changes_heads_freezes_target_and_resumes_exactly(components, tmp_path, monkeypatch):
    from four_anchor import packed_train as pt
    target, _, _, cfg = setup(components)
    original = {k: v.clone() for k, v in target.state_dict().items()}
    data = tmp_path / "data"
    Dataset.from_dict({"input_ids": [list(range(1, 24))] * 4,
        "loss_mask": [[0, 0] + [1] * 21] * 4}).save_to_disk(data)
    monkeypatch.setattr(pt, "load_target", lambda *a: (target, None))
    cfg = replace(cfg, data=str(data), output=str(tmp_path / "full"),
        max_anchors=2, gradient_accumulation=1, packed_execution="cached", warmup_steps=0)
    full_path = pt.train_packed(cfg)
    early_path = pt.train_packed(replace(cfg, output=str(tmp_path / "early"), max_steps=1))
    resumed_path = pt.train_packed(replace(cfg, output=str(tmp_path / "resumed"), resume=early_path))
    full, early, resumed = [load_file(str(Path(p) / "model.safetensors")) for p in (full_path, early_path, resumed_path)]
    assert set(full) == set(resumed)
    assert all(torch.equal(full[k], resumed[k]) for k in full)
    for name in ("markov_head.markov_w1.weight", "markov_head.markov_w2.weight", "confidence_head.proj.weight"):
        assert not torch.equal(full[name], early[name]), name
    assert all(torch.equal(original[k], v) for k, v in target.state_dict().items())
    rows = [json.loads(s) for s in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    assert all(row["first_confidence_bce"] > 0 and row["second_confidence_bce"] > 0 for row in rows)


def test_config_rejects_accidental_dflash_recipe():
    env = {"MODEL": "/model", "DATASET": "/data", "OUTPUT_DIR": "/out"}
    cfg = build_config(env)
    assert cfg.num_anchors == 4 and cfg.num_proposals == 15 and cfg.num_draft_layers == 5
    assert cfg.max_sequence_length == 3072 and cfg.epochs == 5
    with pytest.raises(ValueError, match="packed bonus_addon"):
        replace(cfg, draft_type="dflash")
    assert replace(cfg, device="npu", loss_graph="npu").device == "npu"
    with pytest.raises(ValueError, match="requires an Ascend"):
        replace(cfg, device="cpu", loss_graph="npu")
    with pytest.raises(ValueError, match="own FourAnchorDSpark"):
        replace(cfg, initialize_dflash="/old")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_dspark_fsdp_state_api_and_optimizer_continuation(components, tmp_path, dtype):
    import torch.distributed as dist
    import torch.testing._internal.distributed.fake_pg  # noqa: F401
    from four_anchor.fsdp_check import run_check
    # Single-rank in-process API test, not a multi-GPU performance/collective test.
    dist.init_process_group("fake", rank=0, world_size=1)
    try:
        run_check(torch.device("cpu"), tmp_path, dtype, draft_type="dspark")
    finally:
        dist.destroy_process_group()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["draft_type"] == "dspark" and result["checkpoint_continuation"] == "matched"
