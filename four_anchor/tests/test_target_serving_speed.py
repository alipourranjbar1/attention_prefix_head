import json
from pathlib import Path

import pytest

from four_anchor.target_serving import settings
from four_anchor.speed_benchmark import graph_observed, summarize


def test_graph_environment_applies_only_to_teacher_and_covers_prefill():
    parent = {"TARGET_EXECUTION": "graph", "TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1"}
    child, args, record = settings(parent)
    assert parent["TORCH_COMPILE_DISABLE"] == "1"
    assert "TORCH_COMPILE_DISABLE" not in child and "TORCHDYNAMO_DISABLE" not in child
    assert "--enforce-eager" not in args
    config = json.loads(args[args.index("--compilation-config") + 1])
    assert config["mode"] == 3 and config["cudagraph_mode"] == "PIECEWISE"
    assert max(config["cudagraph_capture_sizes"]) == 3072
    assert "--cudagraph-metrics" in args and not record["runtime_graph_verified"]


def test_eager_control_and_bad_sizes():
    _, args, _ = settings({})
    assert args == ["--enforce-eager"]
    for value in ("0", "512,128", "32,32", "10000"):
        with pytest.raises(ValueError):
            settings({"TARGET_EXECUTION": "graph", "TARGET_GRAPH_SIZES": value})


def test_runtime_metric_does_not_accept_config_or_warmup_as_evidence():
    assert graph_observed('"cudagraph_mode":"PIECEWISE"\nCapturing graphs') == 0
    text = "| 253 | 256 | 3 | PIECEWISE | 8 |\n| 2 | 128 | 126 | CUDAGraphMode.PIECEWISE | 1 |"
    assert graph_observed(text) == 9
    assert graph_observed("| 253 | 253 | 0 | NONE | 8 |") == 0


def test_trial_summary_excludes_startup_and_requires_completion(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [{"step": s, "max_rank_step_seconds": 100. if s <= 3 else 7., "loss": 1.,
             "rank0_peak_memory_allocated_gib": 16., **{"max_rank_" + k + "_seconds": .5
                for k in ("input", "forward", "backward", "optimizer")}} for s in range(1, 10)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert summarize(path, 3, 9)["median_step_seconds"] == 7.
    with pytest.raises(ValueError, match="complete"):
        summarize(path, 3, 10)


def test_trials_are_fresh_bounded_and_failed_graph_is_not_selected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import four_anchor.speed_benchmark as module
    monkeypatch.delenv("RESUME", raising=False)
    calls = []
    def execute(command, cwd, env, stdout, stderr):
        calls.append(env.copy())
        assert env["BENCHMARK_ONLY"] == "1" and env["MAX_STEPS"] == "10"
        assert env["MAX_ANCHORS"] == "1536" and env["SEQ_LENGTH"] == "9216"
        out = Path(env["OUTPUT_DIR"])
        (out / "checkpoints").mkdir(parents=True)
        speed = 8. if env["HEAD_LOSS_BACKEND"] == "checkpoint" else 6.
        rows = [{"step": n, "max_rank_step_seconds": speed, "loss": 1., "rank0_peak_memory_allocated_gib": 17.,
                 **{"max_rank_" + k + "_seconds": .5 for k in ("input", "forward", "backward", "optimizer")}}
                for n in range(1, 11)]
        (out / "checkpoints/metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (out / "target_server.log").write_text('"cudagraph_mode": "PIECEWISE"')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", execute)
    output = tmp_path / "runs"
    module.run(tmp_path, output, steps=10, warmup=3)
    assert len(calls) == 9
    selected = json.loads((output / "selected.json").read_text())
    assert selected["settings"]["HEAD_LOSS_BACKEND"] == "cached_vjp"
    assert selected["settings"]["TARGET_EXECUTION"] == "eager"
    assert "MAX_STEPS" not in (output / "selected.env").read_text().split("unset")[0]
    records = json.loads((output / "results.json").read_text())["trials"]
    assert next(r for r in records if r["name"] == "07_target_graph")["status"] == "failed"


def test_failed_reference_stops_before_other_allocations(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import four_anchor.speed_benchmark as module
    monkeypatch.delenv("RESUME", raising=False)
    called = []
    def fail(*args, **kwargs):
        called.append(1)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Reference run failed"):
        module.run(tmp_path, tmp_path / "runs", steps=10, warmup=3)
    assert len(called) == 1
