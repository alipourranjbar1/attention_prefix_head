import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def restore_prepare_environment():
    names = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "TORCH_DEVICE_BACKEND_AUTOLOAD", "USE_TORCH", "USE_TF")
    saved = {name: os.environ.get(name) for name in names}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture
def suite():
    path = Path(__file__).parents[1] / "scripts/run_cached_benchmarks.py"
    spec = importlib.util.spec_from_file_location("cached_benchmark_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recovered_root(suite, tmp_path):
    # This test checkout may contain a NEW runtime; the recovered path must
    # still be byte-identical to the preserved published implementation.
    path = tmp_path / "recovered"
    repo = Path(__file__).parents[2]
    for name in suite.RUNTIME_FILES:
        dest = path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.check_output(["git", "-C", str(repo), "show", f"{suite.GOOD_COMMIT}:{name}"]))
    return path


SAMPLE = {
    "gsm8k": {"question": "How many oranges?", "answer": "SECRET_ANSWER"},
    "math500": {"problem": "Solve x+1=2.", "solution": "SECRET_ANSWER"},
    "aime25": {"problem": "Compute the integer.", "answer": "SECRET_ANSWER"},
    "humaneval": {"prompt": "def f(x):\n    '''Return x.'''", "canonical_solution": "SECRET_ANSWER", "test": "SECRET_TEST"},
    "mbpp": {"prompt": "Write a sum function.", "code": "SECRET_ANSWER", "test_list": ["SECRET_TEST"]},
    "lcb": {"question_content": "Read n and print n.", "starter_code": "def solve():", "private_test_cases": "SECRET_TEST"},
    "mtbench": {"turns": ["Write a poem.", "SECRET_SECOND_TURN"], "reference": "SECRET_ANSWER"},
    "alpaca": {"instruction": "Summarize.", "input": "The public article.", "output": "SECRET_ANSWER"},
}


@pytest.mark.parametrize("name", SAMPLE)
def test_prompts_never_include_reference_answers(suite, name):
    messages = suite.format_messages(name, SAMPLE[name])
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert "SECRET_" not in messages[0]["content"]
    if name in {"gsm8k", "math500", "aime25"}:
        assert messages[0]["content"].endswith("\\boxed{}.")


def test_older_mbpp_and_mtbench(suite):
    assert suite.format_messages("mbpp", {"text": "public", "code": "SECRET"})[0]["content"] == "public"
    assert suite.format_messages("mtbench", {"prompt": "full first turn"})[0]["content"] == "full first turn"
    assert suite.format_messages("mtbench", {"messages": [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "SECRET"},
        {"role": "user", "content": "second"}]})[0]["content"] == "first"


@pytest.mark.parametrize("value", [[1, 2], {"input_ids": [[1, 2]]}, SimpleNamespace(ids=[1, 2])])
def test_token_encodings(suite, value):
    assert suite.token_list(value) == [1, 2]


@pytest.mark.parametrize("value", [[], [[1], [2]], [1, -1], [True, 2]])
def test_invalid_encodings_rejected(suite, value):
    with pytest.raises(ValueError):
        suite.token_list(value)


def write_arrow(folder, rows, split, prefix="data", expected=None):
    import pyarrow as pa
    folder.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    with (folder / f"{prefix}-{split}.arrow").open("wb") as stream:
        with pa.ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    (folder / "dataset_info.json").write_text(json.dumps({"splits": {
        split: {"num_examples": len(rows) if expected is None else expected}}}))


def test_arrow_selects_exact_config_and_split(suite, tmp_path):
    root = tmp_path / "datasets"
    write_arrow(root / "openai___gsm8k/main/0/hash", [{"question": "correct"}], "test", "gsm8k")
    write_arrow(root / "openai___gsm8k/main/0/hash", [{"question": "TRAIN"}], "train", "gsm8k")
    write_arrow(root / "openai___gsm8k/socratic/0/hash", [{"question": "SOCRATIC"}], "test", "gsm8k")
    rows, info = suite.cached_rows("gsm8k", [root], tmp_path / "loader", 5, [])
    assert rows[0]["question"] == "correct"
    assert info["config"] == "main" and info["split"] == "test"


def test_incomplete_arrow_cache_is_not_silently_used(suite, tmp_path):
    root = tmp_path / "datasets"
    write_arrow(root / "openai___gsm8k/main/0/hash", [{"question": "one"}], "test", "gsm8k", expected=2)
    diagnostics = []
    with pytest.raises(ValueError, match="could not be read"):
        suite.cached_rows("gsm8k", [root], tmp_path / "loader", 5, diagnostics)
    assert any("Incomplete test cache" in x for x in diagnostics)


def test_hub_parquet_does_not_mix_mbpp_configs_or_train(suite, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    hub = tmp_path / "hub"
    snapshot = hub / "datasets--google-research-datasets--mbpp/snapshots/rev"
    for config, split, value in [("sanitized", "test", "right"), ("sanitized", "train", "TRAIN"), ("full", "test", "FULL")]:
        directory = snapshot / config
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([{"prompt": value}]), directory / f"{split}-00000-of-00001.parquet")
    rows, _ = suite.cached_rows("mbpp", [hub], tmp_path / "loader", 5, [])
    assert len(rows) == 1 and rows[0]["prompt"] == "right"


def test_lcb_cached_shards_are_explicit(suite, tmp_path):
    snapshot = tmp_path / "hub/datasets--livecodebench--code_generation_lite/snapshots/rev"
    snapshot.mkdir(parents=True)
    for filename in ["test.jsonl", "test3.jsonl"]:
        (snapshot / filename).write_text(json.dumps({"question_content": filename}) + "\n")
    rows, info = suite.cached_rows("lcb", [tmp_path / "hub"], tmp_path / "loader", 5, [])
    assert len(rows) == 2
    assert info["missing_lcb_release_files"] == ["test2.jsonl", "test4.jsonl", "test5.jsonl"]


def test_saved_dataset_dict_uses_selected_split(suite, tmp_path):
    from datasets import Dataset, DatasetDict
    path = tmp_path / "saved"
    DatasetDict({"test": Dataset.from_dict({"prompt": ["test"]}),
                 "train": Dataset.from_dict({"prompt": ["train"]})}).save_to_disk(path)
    rows, _ = suite.load_explicit(path, "test", tmp_path / "loader")
    assert rows[0]["prompt"] == "test"


def test_manifest_tokenizer_identity_and_thinking(suite, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        (model / name).write_text("{}")
    data = {"target": str(model), "thinking": False,
        "tokenizer_sha256": {name: suite.sha(model / name) for name in ["tokenizer.json", "tokenizer_config.json"]},
        "datasets": {"gsm8k": {"results": [{"prompt_token_ids": [1, 2]}]}}}
    manifest = tmp_path / "prompts.json"
    manifest.write_text(json.dumps(data))
    assert len(suite.discover_manifests([manifest], [], model, [])) == 1
    (model / "tokenizer.json").write_text("CHANGED")
    diagnostics = []
    assert suite.discover_manifests([manifest], [], model, diagnostics) == []
    assert "Tokenizer mismatch" in diagnostics[0]
    data["thinking"] = True
    manifest.write_text(json.dumps(data))
    assert suite.discover_manifests([manifest], [], model, []) == []


def test_context_budget_reserves_redraft_positions(suite):
    assert suite.context_length([100, 1900], 2048, 32768) == 4096
    assert suite.context_length([6000], 2048, 32768) == 8192
    with pytest.raises(ValueError, match="No prompts truncated"):
        suite.context_length([6000], 2048, 32768, 4096)


def tiny_model(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    path.mkdir()
    backend = Tokenizer(WordLevel({"[UNK]": 0, "user": 1, "assistant": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}assistant"
    tokenizer.save_pretrained(path)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3", "max_position_embeddings": 32768, "vocab_size": 3}))


def fixture_sources(suite, tmp_path):
    cache = tmp_path / "datasets"
    for name, (repo, config, split) in suite.BENCHMARKS.items():
        folder = cache / repo.replace("/", "___") / config / "0.0.0/hash"
        write_arrow(folder, [SAMPLE[name]] * 3, split)
    return cache


def test_entire_prepare_path_offline_all_eight_benchmarks(suite, tmp_path, monkeypatch, recovered_root):
    import socket
    def deny(*args, **kwargs):
        raise AssertionError("No network allowed in offline preparation")
    monkeypatch.setattr(socket.socket, "connect", deny)
    model = tmp_path / "model"
    tiny_model(model)
    cache = fixture_sources(suite, tmp_path)
    monkeypatch.setattr(suite, "cache_roots", lambda extra: [cache])
    repo = Path(__file__).parents[2]
    output = tmp_path / "suite"
    args = ["--root", str(repo), "--code-root", str(recovered_root), "--model", str(model),
            "--output", str(output), "--limit", "2", "--prepare-only"]
    assert suite.main(args) == 0
    plan = json.loads((output / "plan.json").read_text())
    assert len(plan["benchmarks"]) == 8
    assert all(e["status"] == "prepared" and e["prompts"] == 2 for e in plan["benchmarks"].values())
    for name in suite.BENCHMARKS:
        rows = [json.loads(s) for s in (output / name / "prompts.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(type(t) is int for row in rows for t in row["input_ids"])
    assert "known_bonus_cached_v1" in (output / "suite_summary.json").read_text()


def test_suite_preserves_results_and_continues_after_one_failure(suite, tmp_path, monkeypatch, recovered_root):
    model = tmp_path / "model"
    tiny_model(model)
    cache = fixture_sources(suite, tmp_path)
    monkeypatch.setattr(suite, "cache_roots", lambda extra: [cache])
    monkeypatch.setenv("CONDA_PREFIX", "/dummy/conda")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ["model.safetensors", "four_anchor_config.json"]:
        (checkpoint / name).write_text("fixture; engine is mocked")
    calls = []
    def fake_run(args, code_root, output, name, entry, mode):
        calls.append((name, mode))
        if name == "mbpp":
            raise RuntimeError("synthetic engine failure")
        destination = output / name / mode
        destination.mkdir()
        suite.write_json(destination / "summary.json", {
            "prompts": entry["prompts"], "output_tokens": 120, "seconds": 2,
            "verification_cycles": 20, "accepted_draft_per_verification": 5,
            "acceptance_length_including_bonus": 6, "tokens_per_second_including_prefill": 60})
    monkeypatch.setattr(suite, "run_benchmark", fake_run)
    repo = Path(__file__).parents[2]
    output = tmp_path / "run"
    assert suite.main(["--root", str(repo), "--code-root", str(recovered_root), "--model", str(model),
        "--draft", str(checkpoint), "--output", str(output), "--limit", "2"]) == 1
    report = json.loads((output / "suite_summary.json").read_text())
    assert report["benchmarks"]["mbpp"]["status"] == "failed"
    assert sum(x["status"] == "completed" for x in report["benchmarks"].values()) == 7
    assert len(calls) == 8
    assert report["overall_completed_only"]["tokens_per_second_including_prefill"] == 60
    assert report["overall_completed_only"]["acceptance_length_including_bonus"] == 6


def test_launcher_keeps_recovered_mode_and_restores_backend_env(suite, tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("USE_TORCH", "0")
    entry = {"data": "prompts.jsonl", "prompts": 2, "max_model_len": 4096}
    args = SimpleNamespace(root=str(tmp_path / "dependency"), run=str(tmp_path / "training"),
        model="local_model", draft="checkpoint", npu=1, max_new_tokens=2048, warmup=2, memory_fraction=.8, with_baseline=False)
    output = tmp_path / "results"
    def process(command, cwd, env):
        assert command == ["bash", str(tmp_path / "recovered/four_anchor/scripts/vllm_ascend.sh"), "evaluate"]
        assert env["NUM_SPECULATIVE_TOKENS"] == "15" and env["DRAFT_GRAPH"] == "0"
        assert env["MODE"] == "two_pass" and env["LIMIT"] == "2" and env["EVAL_NPU"] == "1"
        assert "TORCH_DEVICE_BACKEND_AUTOLOAD" not in env and "USE_TORCH" not in env
        destination = Path(env["OUTPUT_DIR"])
        destination.mkdir(parents=True)
        suite.write_json(destination / "summary.json", {"prompts": 2, "mode": "two_pass",
            "inference_version": "known_bonus_cached_v1", "speculative_tokens": 15})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite.subprocess, "run", process)
    suite.run_benchmark(args, tmp_path / "recovered", output, "gsm8k", entry, "two_pass")


def test_predictor_runtime_is_rejected(suite, tmp_path):
    repo = Path(__file__).parents[2]
    with pytest.raises(ValueError, match="differs from the recovered"):
        suite.runtime_checkout(repo, tmp_path, tmp_path / "out")


def test_downloaded_script_prepares_in_fresh_process(suite, tmp_path, recovered_root):
    model = tmp_path / "model"
    tiny_model(model)
    cache = fixture_sources(suite, tmp_path)
    repo = Path(__file__).parents[2]
    script = repo / "four_anchor/scripts/run_cached_benchmarks.py"
    output = tmp_path / "standalone"
    env = os.environ.copy()
    for name in ("USE_TORCH", "USE_TF", "TORCH_DEVICE_BACKEND_AUTOLOAD"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, str(script), "--root", str(repo), "--code-root", str(recovered_root),
        "--model", str(model), "--cache-dir", str(cache), "--output", str(output), "--prepare-only", "--limit", "2"],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads((output / "plan.json").read_text())
    assert all(e["status"] == "prepared" for e in plan["benchmarks"].values())
