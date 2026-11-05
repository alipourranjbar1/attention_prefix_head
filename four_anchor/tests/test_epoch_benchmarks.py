from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

from four_anchor import epoch_benchmarks as eb
from four_anchor.provenance import checkpoint_hashes
from four_anchor.tests.test_cached_benchmarks import suite, tiny_model, fixture_sources, restore_prepare_environment
from four_anchor.tests.test_packed_training import tiny_config


def checkpoint(path):
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"synthetic checkpoint, no hardware engine")
    (path / "four_anchor_config.json").write_text(json.dumps({"draft_config": {
        "num_anchors": 4, "num_proposals": 15, "anchor_layout": "target_four", "format_version": 3}}))
    checkpoint_hashes(path)


def test_jobs_are_complete_idempotent_and_detect_replaced_weights(tmp_path):
    path = tmp_path / "checkpoint"
    checkpoint(path)
    eb.publish_epoch(tmp_path / "evaluation", path, 1, 10)
    eb.publish_epoch(tmp_path / "evaluation", path, 1, 10)
    job = json.loads((tmp_path / "evaluation/queue/epoch-001.json").read_text())
    eb.validate_checkpoint(job)
    with pytest.raises(ValueError, match="different checkpoint"):
        eb.publish_epoch(tmp_path / "evaluation", path, 1, 11)
    (path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        eb.validate_checkpoint(job)


def test_new_prepare_and_exact_prompt_reuse_for_all_eight_benchmarks(suite, tmp_path, monkeypatch):
    model = tmp_path / "model"
    tiny_model(model)
    cache = fixture_sources(suite, tmp_path)
    repo = Path(__file__).parents[2]
    monkeypatch.setattr(suite, "cache_roots", lambda _: [cache])
    common = ["--root", str(repo), "--code-root", str(repo), "--inference-layout", "target_four",
              "--model", str(model), "--prepare-only", "--limit", "2"]
    frozen = tmp_path / "frozen"
    assert suite.main(common + ["--output", str(frozen)]) == 0
    output = tmp_path / "epoch1"
    assert suite.main(common + ["--output", str(output), "--prompt-plan", str(frozen / "plan.json")]) == 0
    plan = json.loads((output / "plan.json").read_text())
    assert plan["query_positions"] == 19 and plan["method"] == "target_four_materialized_v3"
    assert len(plan["benchmarks"]) == 8
    for name in plan["benchmarks"]:
        assert (output / name / "prompts.jsonl").read_bytes() == (frozen / name / "prompts.jsonl").read_bytes()
    assert b"\r" not in (output / "summary.csv").read_bytes()
    (frozen / "gsm8k/prompts.jsonl").write_text('{"input_ids":[5]}\n')
    with pytest.raises(ValueError, match="prompts changed"):
        suite.main(common + ["--output", str(tmp_path / "bad"), "--prompt-plan", str(frozen / "plan.json")])


def test_worker_preserves_failed_epoch_and_continues_with_next(tmp_path, monkeypatch):
    directory = tmp_path / "evaluation"
    (directory / "prepared").mkdir(parents=True)
    (directory / "prepared/plan.json").write_text("{}")
    for epoch in (1, 2):
        path = tmp_path / f"checkpoint{epoch}"
        checkpoint(path)
        eb.publish_epoch(directory, path, epoch, epoch * 10)
    calls = []
    class Child:
        def __init__(self, command, stdout, stderr):
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            self.returncode = 1 if len(calls) == 1 else 0
            eb.write_json(output / "suite_summary.json", {"benchmarks": {"gsm8k": {
                "status": "failed" if self.returncode else "completed", "prompts": 128,
                "summary": {"prompts": 128, "acceptance_length_including_bonus": 6.2,
                            "tokens_per_second_including_prefill": 50.1}}}})
        def poll(self):
            return self.returncode
    monkeypatch.setattr(eb.subprocess, "Popen", Child)
    args = SimpleNamespace(directory=str(directory), root="deps", code_root="code", run="run", model="model",
                           npu=7, limit=128, max_new_tokens=2048, once=True, retry_failed=False)
    assert eb.worker(args) == 1
    assert len(calls) == 2
    for c in calls:
        assert c[c.index("--inference-layout") + 1] == "target_four"
        assert c[c.index("--npu") + 1] == "7"
    table = (directory / "epoch_results.md").read_text()
    assert "| 1 | 10 | gsm8k | failed |" in table
    assert "| 2 | 20 | gsm8k | completed |" in table
    assert b"\r" not in (directory / "epoch_results.csv").read_bytes()
    assert eb.worker(args) == 1
    assert len(calls) == 2  # no duplicate work on a normal restart


def test_trainer_queues_only_complete_epochs_not_partial_steps(components, tmp_path, monkeypatch):
    from four_anchor import packed_train as pt
    target, _, _ = components
    data = tmp_path / "data"
    Dataset.from_dict({"input_ids": [list(range(1, 24))] * 3,
                      "loss_mask": [[0, 0] + [1] * 21] * 3}).save_to_disk(data)
    monkeypatch.setattr(pt, "load_target", lambda *a: (target, None))
    cfg = replace(tiny_config(num_proposals=15, anchor_layout="target_four"),
        data=str(data), output=str(tmp_path / "checkpoints"), gradient_checkpointing=False,
        checkpoint_each_epoch=True, epoch_benchmark_dir=str(tmp_path / "eval"),
        max_steps=4, max_anchors=2, blocks_per_chunk=2, head_chunk_size=16, gradient_accumulation=1)
    saved = pt.train_packed(cfg)
    jobs = list((tmp_path / "eval/queue").glob("*.json"))
    assert len(jobs) == 1  # three updates in epoch 1, then one partial epoch-2 update
    job = json.loads(jobs[0].read_text())
    assert (job["epoch"], job["step"]) == (1, 3)
    eb.validate_checkpoint(job)
    assert Path(saved).name == "checkpoint-000004"
    # Resume completes epoch 2 and publishes precisely one additional job.
    cfg = replace(cfg, resume=saved, max_steps=None)
    pt.train_packed(cfg)
    jobs = sorted((tmp_path / "eval/queue").glob("*.json"))
    assert len(jobs) == 2
    assert json.loads(jobs[1].read_text())["epoch"] == 2


def test_new_launcher_clears_training_distributed_identity(suite, tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("WORLD_SIZE", "5")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "trainer")
    args = SimpleNamespace(root=str(tmp_path / "deps"), run="run", model="model", draft="draft",
        npu=7, max_new_tokens=2048, warmup=2, memory_fraction=.8, with_baseline=False,
        inference_layout="target_four")
    def launch(command, cwd, env):
        assert "RANK" not in env and "WORLD_SIZE" not in env and "TORCHELASTIC_RUN_ID" not in env
        assert env["EVAL_NPU"] == "7" and env["NUM_SPECULATIVE_TOKENS"] == "15"
        folder = Path(env["OUTPUT_DIR"])
        folder.mkdir(parents=True)
        suite.write_json(folder / "summary.json", {"prompts": 2, "mode": "two_pass",
            "inference_version": "target_four_materialized_v3", "speculative_tokens": 15})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite.subprocess, "run", launch)
    suite.run_benchmark(args, tmp_path / "code", tmp_path / "out", "gsm8k",
                       {"data": "p.jsonl", "prompts": 2, "max_model_len": 4096}, "two_pass")
