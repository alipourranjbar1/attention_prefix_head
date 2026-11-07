import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from four_anchor.tests.test_accepted_prefix import target19
from four_anchor.tests.test_cached_benchmarks import tiny_model, fixture_sources, restore_prepare_environment
from four_anchor import vllm_batch_entry as entry


def args(checkpoint):
    return NS(target="tiny", checkpoint=str(checkpoint), mode="two_pass", proposals=15,
        concurrency=32, max_model_len=128, memory_fraction=.8, anchor_policy="bonus_addon",
        draft_graph=False, profile_steps=0, trace_steps=False, data="local", output=None,
        limit=128, max_new_tokens=16, warmup=1, baseline=None)


def test_engine_has_one_worker_and_native_limit_32(components, tmp_path):
    target19(components).save(tmp_path)
    opts = entry.engine_options(args(tmp_path))
    assert opts["max_num_seqs"] == 32
    assert opts["speculative_config"]["num_speculative_tokens"] == 15
    assert opts["worker_cls"] == "four_anchor.vllm_batch_ascend.BatchedFourAnchorWorker"
    assert opts["tensor_parallel_size"] == 1 and opts["enforce_eager"]
    assert not opts["enable_prefix_caching"] and not opts["async_scheduling"]
    a = args(tmp_path); a.mode = "ar"
    assert "speculative_config" not in entry.engine_options(a)
    assert entry.engine_options(a)["per_request_spec_decode_metrics"] == "none"


def test_evaluator_submits_128_together_not_128_serial_calls(monkeypatch, tmp_path):
    from four_anchor import data
    import transformers
    records = [{"input_ids": [i+1, 2]} for i in range(128)]
    monkeypatch.setattr(data, "load_records", lambda _: records)
    monkeypatch.setattr(data, "prompt_tokens", lambda r, _: r["input_ids"])
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    calls = []
    class LLM:
        def __init__(self, **kwargs): pass
        def collective_rpc(self, method):
            if method == "four_anchor_counters":
                return [{"extra_target_forwards": 0, "bonus_page_fallbacks": 0}]
            if method == "four_anchor_batch_stats":
                return [{"draft_batches": 4, "draft_requests": 128, "max_draft_batch": 32}]
            return [None]
        def generate(self, prompts, params, use_tqdm):
            calls.append(len(prompts))
            m = NS(to_dict=lambda: {"num_spec_tokens": 15, "num_spec_steps": 2, "num_accepted_draft_tokens": 4,
                                  "per_step_accepted": [2, 2]})
            return [NS(prompt_token_ids=p["prompt_token_ids"], outputs=[NS(token_ids=[5]*6, spec_decode_metrics=m)],
                       metrics=NS(arrival_time=0, finished_time=9)) for p in prompts]
    mod = ModuleType("vllm"); mod.LLM = LLM; mod.SamplingParams = lambda **k: NS(**k)
    monkeypatch.setitem(sys.modules, "vllm", mod)
    clocks = iter([100, 110]); monkeypatch.setattr(entry.time, "perf_counter", lambda: next(clocks))
    a = args(tmp_path); a.output = str(tmp_path / "results")
    entry.evaluate(a, {})
    assert calls == [32, 128]
    report = json.loads((Path(a.output) / "summary.json").read_text())
    assert report["seconds"] == 10 and report["tokens_per_second_including_prefill"] == 76.8
    assert report["acceptance_length_including_bonus"] == 3
    assert report["max_draft_batch"] == 32
    assert report["parity_checked"] is False and "measured_speedup_vs_ar" not in report


@pytest.fixture
def batch_suite():
    path = Path(__file__).parents[1] / "scripts/run_batched_benchmarks.py"
    spec = importlib.util.spec_from_file_location("batch_suite", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_cached_suite_reuses_exact_prompts_and_records_concurrency(batch_suite, tmp_path, monkeypatch):
    model = tmp_path / "target"; tiny_model(model)
    cache = fixture_sources(batch_suite, tmp_path)
    repo = Path(__file__).parents[2]
    monkeypatch.setattr(batch_suite, "cache_roots", lambda _: [cache])
    common = ["--root", str(repo), "--code-root", str(repo), "--model", str(model),
              "--prepare-only", "--limit", "2", "--concurrency", "32"]
    frozen, output = tmp_path / "frozen", tmp_path / "new"
    assert batch_suite.main(common + ["--output", str(frozen)]) == 0
    before = {str(p.relative_to(frozen)): p.read_bytes() for p in frozen.rglob("*") if p.is_file()}
    assert batch_suite.main(common + ["--output", str(output), "--prompt-plan", str(frozen / "plan.json")]) == 0
    assert before == {str(p.relative_to(frozen)): p.read_bytes() for p in frozen.rglob("*") if p.is_file()}
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "suite_summary.json").read_text())
    assert plan["concurrency"] == report["concurrency"] == 32
    assert len(plan["benchmarks"]) == 8
    assert plan["query_positions"] == {"round_one": 20, "round_two": 19}
    assert "Concurrency limit: 32" in (output / "summary.md").read_text()


def test_suite_launches_batch_entry_with_concurrency(batch_suite, tmp_path, monkeypatch):
    root = Path(__file__).parents[2]
    out = tmp_path / "out"; folder = out / "gsm8k/two_pass"; folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"prompts": 128, "mode": "two_pass", "concurrency": 32,
        "inference_version": "bonus_addon20_draft4_batch_v1", "speculative_tokens": 15}))
    seen = {}
    def run(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs); return NS(returncode=0)
    monkeypatch.setattr(batch_suite.subprocess, "run", run)
    a = NS(root=str(root), run="run", model="model", draft="checkpoint", concurrency=32,
        inference_layout="bonus_addon_batch", with_baseline=False, npu=7, max_new_tokens=2048, warmup=2, memory_fraction=.8)
    batch_suite.run_benchmark(a, root, out, "gsm8k", {"data": "frozen", "prompts": 128, "max_model_len": 4096}, "two_pass")
    assert seen["env"]["CONCURRENCY"] == "32" and seen["env"]["EVAL_NPU"] == "7"
    assert seen["env"]["ANCHOR_POLICY"] == "bonus_addon"
    assert seen["cmd"][1].endswith("vllm_batch_ascend.sh")
