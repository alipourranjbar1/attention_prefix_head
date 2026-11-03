#!/usr/bin/env python3
"""Offline benchmark suite for the recovered four-anchor embedding-only method.

Run after activating attention_prefix_ali:
  python run_cached_benchmarks.py --npu 1 --limit 128
  python run_cached_benchmarks.py --prepare-only

Discovers compatible Evaluator prompt manifests, HF Arrow caches, local HF Hub
snapshots, and save_to_disk datasets. No remote dataset IDs are loaded or fetched.
Use --cache-dir PATH, --manifest FILE, or --source NAME=PATH for other locations.
--limit 0 uses all rows in the selected cached source (a manifest may be a subset).
MT-Bench uses the first user turn. Reports speed/acceptance, not task accuracy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


GOOD_COMMIT = "9a2671f3283015952c73265c6bf3c3741c22a045"
DEFAULT_ROOT = "/home/n84449292/a84384644/attention_head_forked/four-anchor-dflash-pipeline"
DEFAULT_MODEL = "/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
BENCHMARKS = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "default", "test"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test"),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test"),
    "lcb": ("livecodebench/code_generation_lite", "default", "test"),
    "aime25": ("MathArena/aime_2025", "default", "train"),
    "mtbench": ("HuggingFaceH4/mt_bench_prompts", "default", "train"),
    "alpaca": ("tatsu-lab/alpaca", "default", "train"),
}
ALIASES = {"math-500": "math500", "mt-bench": "mtbench", "aime2025": "aime25",
           "livecodebench": "lcb", "human_eval": "humaneval"}
RUNTIME_FILES = ["four_anchor/" + name for name in (
    "model.py", "data.py", "inference.py", "vllm_support.py", "vllm_ascend.py",
    "vllm_entry.py", "scripts/vllm_ascend.sh")]
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def canonical(name):
    return ALIASES.get(name.lower(), name.lower())


def token_list(value):
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("Expected one nonempty list of integer token IDs")
    return list(value)


def format_messages(name, row):
    """Select only public problem fields; never feed answers/solutions to the model."""
    if name in {"gsm8k", "math500", "aime25"}:
        question = row.get("question") if name == "gsm8k" else row.get("problem", row.get("question"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{name}: missing question/problem")
        text = question + MATH_SUFFIX
    elif name == "humaneval":
        text = ("Write a solution to the following problem and make sure that it passes the tests:\n```python\n"
                + row["prompt"] + "\n```")
    elif name == "mbpp":
        # HF sanitized MBPP uses 'prompt'; full/older local variants may use 'text'.
        text = row.get("prompt") or row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("mbpp: missing public prompt/text")
    elif name == "lcb":
        starter = row.get("starter_code") or ""
        text = (row["question_content"] + "\n\n" + (
            "Complete the following Python starter code.\n```python\n" + starter + "\n```" if starter else
            "Write a Python program that reads from standard input and writes to standard output.")
            + "\nReturn your solution in a Python code block.")
    elif name == "alpaca":
        text = row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
    elif name == "mtbench":
        turns = row.get("turns") or row.get("prompt")
        if isinstance(turns, (list, tuple)):
            text = turns[0]
        elif isinstance(turns, str):
            text = turns
        else:
            # Only the first user message, never assistant/reference responses.
            text = next((x["content"] for x in row.get("messages", []) if x.get("role") == "user"), None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("mtbench: missing first user turn")
    else:
        raise ValueError(name)
    return [{"role": "user", "content": text}]


def unique_paths(paths):
    return list(dict.fromkeys(Path(p).expanduser().resolve() for p in paths if p))


def cache_roots(extra):
    home = Path.home()
    hf = Path(os.environ.get("HF_HOME", home / ".cache/huggingface"))
    custom = Path("/home/n84449292/m84379596/Huggingface")
    raw = [*extra, os.environ.get("HF_DATASETS_CACHE"), os.environ.get("HF_HUB_CACHE"),
           hf, hf / "datasets", hf / "hub", custom, custom / "datasets", custom / "hub"]
    paths = unique_paths(raw)
    return unique_paths([*paths, *(p / sub for p in paths for sub in ("datasets", "hub"))])


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def read_files(paths, cache):
    from datasets import Dataset, concatenate_datasets
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("No local data shards")
    if paths[0].suffix == ".arrow":
        return concatenate_datasets([Dataset.from_file(str(p)) for p in paths])
    if paths[0].suffix == ".parquet":
        return Dataset.from_parquet([str(p) for p in paths], cache_dir=str(cache))
    return Dataset.from_json([str(p) for p in paths], cache_dir=str(cache))


def load_explicit(path, split, cache):
    from datasets import DatasetDict, load_from_disk
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        rows = load_from_disk(str(path))
        if isinstance(rows, DatasetDict):
            rows = rows[split]
    else:
        rows = read_files([path], cache)
    return rows, {"kind": "explicit_local_source", "path": str(path), "split": split}


def cached_rows(name, roots, cache, lcb_release, diagnostics):
    """Read local cache bytes directly; never call load_dataset with a remote ID."""
    repo, config, split = BENCHMARKS[name]
    labels = {normalized(name), normalized(repo), normalized(repo.split("/")[-1])}
    directories = []
    for root in roots:
        if root.is_dir():
            directories += [p for p in root.iterdir() if p.is_dir() and normalized(p.name) in labels]
            if normalized(root.name) in labels:
                directories.append(root)
    for folder in unique_paths(directories):
        if (folder / "state.json").exists() or (folder / "dataset_dict.json").exists():
            try:
                return load_explicit(folder, split, cache)
            except Exception as exc:
                diagnostics.append(f"{folder}: {type(exc).__name__}: {exc}")
        config_dirs = [p for p in folder.iterdir() if p.is_dir() and p.name.lower() == config.lower()]
        # Match the exact dataset configuration, especially main vs socratic and sanitized vs full.
        infos = [p for d in config_dirs for p in d.glob("**/dataset_info.json")]
        for info in sorted(infos, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(info.read_text())
                shards = sorted(p for p in info.parent.glob("*.arrow")
                                if re.search(r"-" + re.escape(split) + r"(?:-\d+-of-\d+)?\.arrow$", p.name))
                if not shards:
                    continue
                rows = read_files(shards, cache)
                expected = meta.get("splits", {}).get(split, {}).get("num_examples")
                if expected is not None and len(rows) != expected:
                    raise ValueError(f"Incomplete {split} cache: {len(rows)} of {expected} rows")
                return rows, {"kind": "hf_arrow_cache", "repo": repo, "config": config, "split": split,
                              "path": str(info.parent), "files": [str(p) for p in shards]}
            except Exception as exc:
                diagnostics.append(f"{info}: {type(exc).__name__}: {exc}")
    hub_name = "datasets--" + repo.replace("/", "--")
    hub_dirs = []
    for root in roots:
        if root.is_dir():
            hub_dirs += [p for p in root.iterdir() if p.is_dir() and p.name.lower() == hub_name.lower()]
            if root.name.lower() == hub_name.lower():
                hub_dirs.append(root)
    for hub in unique_paths(hub_dirs):
        snapshots = list((hub / "snapshots").glob("*"))
        for snap in sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                if name == "lcb":
                    expected_files = ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, lcb_release + 1)]
                    files = [snap / f for f in expected_files if (snap / f).is_file()]
                    if not files:
                        continue
                    missing = [f for f in expected_files if not (snap / f).is_file()]
                else:
                    location = snap / config if (snap / config).is_dir() else snap
                    files = sorted(p for p in location.rglob("*.parquet") if p.is_file() and
                                   re.search(r"(?:^|[/_-])" + split + r"(?:[/_.-]|$)", str(p.relative_to(location))))
                    if not files:
                        files = sorted(p for p in location.rglob("*") if p.is_file() and p.suffix in {".jsonl", ".json"}
                                       and re.search(r"(?:^|[/_-])" + split + r"(?:[/_.-]|$)", str(p.relative_to(location))))
                    if not files:
                        continue
                    missing = []
                return read_files(files, cache), {"kind": "hf_hub_snapshot", "repo": repo, "config": config,
                    "split": split, "path": str(snap), "files": [str(p) for p in files],
                    "missing_lcb_release_files": missing, "requested_lcb_release": lcb_release if name == "lcb" else None}
            except Exception as exc:
                diagnostics.append(f"{snap}: {type(exc).__name__}: {exc}")
    if diagnostics:
        raise ValueError(f"Cached {repo}/{config}/{split} files could not be read; see diagnostics")
    raise FileNotFoundError(f"No readable cached {repo}/{config}/{split}")


def discover_manifests(explicit, roots, model, diagnostics):
    paths = list(explicit)
    for root in roots:
        folder = root / "Evaluator"
        if folder.is_dir():
            paths += list(folder.glob("*prompts*.json"))
    model = Path(model).resolve()
    file_hashes = {}
    manifests = []
    for path in unique_paths(paths):
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("datasets"), dict):
                continue
            if data.get("thinking") is not False:
                raise ValueError("Requires an explicitly non-thinking manifest")
            hashes = data.get("tokenizer_sha256") or {}
            if hashes:
                if not {"tokenizer.json", "tokenizer_config.json"} <= hashes.keys():
                    raise ValueError("Manifest lacks tokenizer identity hashes")
                for name, expected in hashes.items():
                    if name not in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"}:
                        raise ValueError("Unexpected tokenizer filename")
                    if name not in file_hashes:
                        file_hashes[name] = sha(model / name)
                    if file_hashes[name] != expected:
                        raise ValueError("Tokenizer mismatch: " + name)
            elif Path(data.get("target", "")).resolve() != model:
                raise ValueError("Manifest has no verifiable matching target/tokenizer")
            config_hash = data.get("target_config_sha256")
            if config_hash:
                if "config.json" not in file_hashes:
                    file_hashes["config.json"] = sha(model / "config.json")
                if file_hashes["config.json"] != config_hash:
                    raise ValueError("Target configuration mismatch")
            manifests.append((path, data))
        except Exception as exc:
            diagnostics.append(f"Manifest {path}: {type(exc).__name__}: {exc}")
    return manifests


def frozen_rows(name, manifests):
    candidates = []
    for path, data in manifests:
        entry = data["datasets"].get(name)
        if entry and isinstance(entry.get("results"), list) and entry["results"]:
            candidates.append((path, entry))
    if not candidates:
        return None
    path, entry = max(candidates, key=lambda item: len(item[1]["results"]))
    return entry["results"], {"kind": "frozen_prompt_manifest", "path": str(path), "sha256": sha(path),
        "source": entry.get("source"), "source_rows": entry.get("source_rows"),
        "manifest_rows": len(entry["results"]), "selection": "existing frozen order"}


def runtime_checkout(root, explicit, output):
    expected = {f: subprocess.check_output(["git", "-C", str(root), "show", f"{GOOD_COMMIT}:{f}"])
                for f in RUNTIME_FILES}
    def matches(path):
        return all((path / f).is_file() and (path / f).read_bytes() == data for f, data in expected.items())
    if explicit:
        path = Path(explicit).resolve()
        if not matches(path):
            raise ValueError("--code-root differs from the recovered embedding-only runtime")
        return path
    candidates = sorted(root.parent.glob("four-anchor-recovery_*"), reverse=True)
    for path in [*candidates, root]:
        if matches(path):
            return path
    path = root.parent / ("four-anchor-recovery_suite_" + output.name)
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", str(path), GOOD_COMMIT], check=True)
    if not matches(path):
        raise RuntimeError("Recovered checkout validation failed")
    return path


def context_length(prompt_lengths, generation, target_limit, override=None):
    required = max(prompt_lengths) + generation + 32  # reserve the redraft position window
    length = override if override else max(4096, math.ceil(required / 128) * 128)
    if length < required or length > target_limit:
        raise ValueError(f"Context needs {required} positions; selected {length}; target limit {target_limit}. No prompts truncated.")
    return length


def prepare(args, output, code_root, names):
    from transformers import AutoTokenizer
    diagnostics = []
    roots = cache_roots(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    target_cfg = json.loads((Path(args.model) / "config.json").read_text())
    target_limit = target_cfg.get("text_config", target_cfg)["max_position_embeddings"]
    origins = [code_root, Path(args.root), Path(args.root).parent / "attention_prefix_head",
               Path("/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main")]
    manifests = discover_manifests(args.manifest, origins, args.model, diagnostics)
    sources = {}
    for entry in args.source:
        key, sep, path = entry.partition("=")
        key = canonical(key)
        if not sep or key not in BENCHMARKS or key in sources:
            raise ValueError("Use unique --source BENCHMARK=LOCAL_PATH mappings")
        sources[key] = path
    plan = {"method": "known_bonus_cached_v1", "code_root": str(code_root), "base_commit": GOOD_COMMIT,
        "runtime_sha256": {f: sha(code_root / f) for f in RUNTIME_FILES}, "target": str(Path(args.model).resolve()),
        "checkpoint": str(Path(args.draft).resolve()), "requested_benchmarks": names,
        "limit_per_benchmark": args.limit, "selection": "first cached rows or existing frozen manifest order",
        "npu": args.npu, "concurrency": 1, "speculative_tokens": 15, "conditioning_anchors": 4,
        "query_positions": 20, "draft_graph": False, "target_graph": False,
        "max_new_tokens": args.max_new_tokens, "thinking": False, "mtbench": "first user turn only",
        "offline": True, "with_baseline": args.with_baseline,
        "cache_roots": [str(p) for p in roots], "benchmarks": {}, "diagnostics": diagnostics}
    for name in names:
        attempts = []
        try:
            frozen = frozen_rows(name, manifests)
            if name in sources:
                rows, info = load_explicit(sources[name], BENCHMARKS[name][2], output / "loader_cache")
            elif frozen and args.limit != 0:
                rows, info = frozen
            else:
                try:
                    rows, info = cached_rows(name, roots, output / "loader_cache", args.lcb_release, attempts)
                except FileNotFoundError:
                    if not frozen:
                        raise
                    rows, info = frozen
            count = min(len(rows), args.limit) if args.limit else len(rows)
            if count == 0:
                raise ValueError("Cached source is empty")
            lengths, prepared = [], []
            for i in range(count):
                row = rows[i]
                if info["kind"] == "frozen_prompt_manifest":
                    ids = token_list(row["prompt_token_ids"])
                else:
                    ids = token_list(tokenizer.apply_chat_template(format_messages(name, row), tokenize=True,
                                    add_generation_prompt=True, enable_thinking=False, return_dict=False))
                if max(ids) >= target_cfg.get("text_config", target_cfg)["vocab_size"]:
                    raise ValueError("Prompt token outside target vocabulary")
                prepared.append({"input_ids": ids, "source_row": row.get("source_row", i),
                                 "source_id": str(row.get("id", row.get("task_id", row.get("question_id", i))))})
                lengths.append(len(ids))
            max_len = context_length(lengths, args.max_new_tokens, target_limit, args.max_model_len)
            folder = output / name
            folder.mkdir()
            prompts = folder / "prompts.jsonl"
            with prompts.open("x") as stream:
                for row in prepared:
                    stream.write(json.dumps(row) + "\n")
            plan["benchmarks"][name] = {"status": "prepared", "prompts": count, "available_rows": len(rows),
                "source": info, "data": str(prompts), "prompt_file_sha256": sha(prompts),
                "max_prompt_tokens": max(lengths), "max_model_len": max_len, "diagnostics": attempts}
            print(f"Prepared {name}: {count}/{len(rows)} cached prompts; max_model_len={max_len}; source={info['kind']}", flush=True)
            if info.get("missing_lcb_release_files"):
                print("  LCB uses available cached shards only; missing: " + ", ".join(info["missing_lcb_release_files"]), flush=True)
            if info.get("source_rows", len(rows)) and info.get("source_rows", len(rows)) > len(rows):
                print(f"  Frozen cache contains {len(rows)} of {info['source_rows']} original dataset rows.", flush=True)
        except FileNotFoundError as exc:
            plan["benchmarks"][name] = {"status": "unavailable", "reason": str(exc), "diagnostics": attempts}
            print(f"Unavailable {name}: {exc}", flush=True)
        except Exception as exc:
            plan["benchmarks"][name] = {"status": "preparation_failed", "reason": f"{type(exc).__name__}: {exc}", "diagnostics": attempts}
            print(f"Preparation failed for {name}: {exc}", flush=True)
        write_json(output / "plan.json", plan)
    return plan


def write_report(output, plan):
    report = {"inference_version": "known_bonus_cached_v1", "concurrency": 1,
              "note": "Acceptance before EOS/budget truncation; throughput includes prefill, excludes loading/warmup. Accuracy not measured.",
              "benchmarks": {}}
    table = []
    successful = []
    for name, entry in plan["benchmarks"].items():
        data = dict(entry)
        summary_path = output / name / "two_pass/summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            data["summary"] = summary
            if entry["status"] == "completed":
                successful.append(summary)
        else:
            summary = {}
        report["benchmarks"][name] = data
        table.append({"benchmark": name, "status": entry["status"], "prompts": summary.get("prompts", entry.get("prompts", 0)),
            "acceptance_including_bonus": summary.get("acceptance_length_including_bonus"),
            "tokens_per_second": summary.get("tokens_per_second_including_prefill"),
            "speedup_vs_ar": summary.get("measured_speedup_vs_ar"),
            "parity": summary.get("all_match_greedy_target"), "output_tokens": summary.get("output_tokens"),
            "seconds": summary.get("seconds"), "reason": entry.get("reason", "")})
    if successful:
        tokens = sum(s["output_tokens"] for s in successful)
        seconds = sum(s["seconds"] for s in successful)
        cycles = sum(s["verification_cycles"] for s in successful)
        accepted = sum(s["accepted_draft_per_verification"] * s["verification_cycles"] for s in successful if s["verification_cycles"])
        report["overall_completed_only"] = {"prompts": sum(s["prompts"] for s in successful),
            "output_tokens": tokens, "seconds": seconds, "verification_cycles": cycles,
            "tokens_per_second_including_prefill": tokens / seconds,
            "acceptance_length_including_bonus": 1 + accepted / cycles if cycles else None}
    write_json(output / "suite_summary.json", report)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    def value(x):
        return "—" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = ["# Cached benchmark results", "", report["note"], "",
        "| Benchmark | Status | Prompts | Acceptance + bonus | Tokens/s | Speedup vs AR | Exact parity |",
        "|---|---|---:|---:|---:|---:|---|"]
    for row in table:
        lines.append("| " + " | ".join(value(row[k]) for k in ("benchmark", "status", "prompts", "acceptance_including_bonus",
                                                            "tokens_per_second", "speedup_vs_ar", "parity")) + " |")
    lines += ["", "Sources, unavailable caches, preparation errors and per-benchmark summaries: `suite_summary.json`.",
              "MT-Bench uses first user turns; a frozen manifest may contain only a cached subset.",
              "No new speedup or parity claim is available unless --with-baseline was used.", ""]
    (output / "summary.md").write_text("\n".join(lines))


def run_benchmark(args, code_root, output, name, entry, mode):
    folder = output / name / mode
    env = os.environ.copy()
    # Preparation disables backend autoload; the launcher initializes CANN before training libraries load.
    for key in ("TORCH_DEVICE_BACKEND_AUTOLOAD", "USE_TORCH", "USE_TF"):
        env.pop(key, None)
    root = Path(args.root).resolve()
    env.update(PYTHONPATH=f"{root}/speculators/src:{root}/speculators/hs_connectors/src",
        VLLM_SOURCE_ROOT=str(root), RUN=str(Path(args.run).resolve()), MODEL=str(Path(args.model).resolve()),
        DRAFT=str(Path(args.draft).resolve()), DATA=entry["data"], OUTPUT_DIR=str(folder),
        MODE=mode, BASELINE=str(output / name / "ar/per_prompt.jsonl") if args.with_baseline and mode != "ar" else "",
        NUM_SPECULATIVE_TOKENS="15", EVAL_NPU=str(args.npu), LIMIT=str(entry["prompts"]),
        MAX_NEW_TOKENS=str(args.max_new_tokens), MAX_MODEL_LEN=str(entry["max_model_len"]),
        DRAFT_GRAPH="0", TRACE_STEPS="0", PROFILE_STEPS="0", WARMUP=str(args.warmup),
        MEMORY_FRACTION=str(args.memory_fraction), PYTHONUNBUFFERED="1")
    print(f"\nRunning {name} / {mode}: {entry['prompts']} prompts on physical NPU {args.npu}", flush=True)
    result = subprocess.run(["bash", str(code_root / "four_anchor/scripts/vllm_ascend.sh"), "evaluate"],
                            cwd=code_root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{mode} exited {result.returncode}; see {folder}/eval.log")
    summary = json.loads((folder / "summary.json").read_text())
    if summary["prompts"] != entry["prompts"] or summary.get("mode") != mode:
        raise RuntimeError("Incomplete or wrong-mode evaluation summary")
    if mode != "ar" and (summary.get("inference_version") != "known_bonus_cached_v1" or summary.get("speculative_tokens") != 15):
        raise RuntimeError("Unexpected inference variant in results")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--code-root", help="Optional recovered checkout; runtime files must match the saved version")
    parser.add_argument("--run")
    parser.add_argument("--draft")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output")
    parser.add_argument("--npu", type=int, default=1)
    parser.add_argument("--limit", type=int, default=128, help="Maximum prompts per dataset; 0 uses every row in its cached source")
    parser.add_argument("--datasets", default="all", help="all or comma-separated names")
    parser.add_argument("--cache-dir", action="append", default=[])
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--lcb-release", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, help="Otherwise sized to fit each benchmark without truncating prompts")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--with-baseline", action="store_true", help="Also run matching AR prompts and check exact output agreement")
    parser.add_argument("--prepare-only", action="store_true", help="Discover/tokenize/cache the prompt plan without using an NPU")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_new_tokens < 1 or args.warmup < 0 or not 0 < args.memory_fraction < 1 or not 1 <= args.lcb_release <= 10 or args.npu < 0:
        parser.error("Invalid numerical option")
    names = list(BENCHMARKS) if args.datasets == "all" else [canonical(n.strip()) for n in args.datasets.split(",")]
    if not names or any(n not in BENCHMARKS for n in names) or len(set(names)) != len(names):
        parser.error("Choose distinct benchmarks from " + ", ".join(BENCHMARKS))
    root = Path(args.root).expanduser().resolve()
    args.root = str(root)
    args.run = args.run or str(root / "output/four_anchor_dflash_20261031_025300_2111006")
    args.draft = args.draft or str(Path(args.run) / "checkpoints/checkpoint-016699")
    args.model = str(Path(args.model).expanduser().resolve())
    if not args.prepare_only:
        if not os.environ.get("CONDA_PREFIX"):
            raise ValueError("Activate attention_prefix_ali before running")
        for name in ("model.safetensors", "four_anchor_config.json"):
            if not (Path(args.draft) / name).is_file():
                raise FileNotFoundError(Path(args.draft) / name)
    output = Path(args.output or (Path(args.run) / "evaluation" / (
        "cached_suite_" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TORCH_DEVICE_BACKEND_AUTOLOAD="0", USE_TORCH="0", USE_TF="0")
    code_root = runtime_checkout(root, args.code_root, output)
    print("Recovered code:", code_root, "\nResults:", output, flush=True)
    plan = prepare(args, output, code_root, names)
    write_report(output, plan)
    ready = [n for n in names if plan["benchmarks"][n]["status"] == "prepared"]
    if not ready:
        print("No benchmarks prepared. Inspect plan.json; use --cache-dir, --manifest or --source NAME=PATH.", flush=True)
        return 2
    if args.prepare_only:
        print("Prepared only; no NPU server started.", flush=True)
        return 1 if any(e["status"] == "preparation_failed" for e in plan["benchmarks"].values()) else 0
    try:
        for i, name in enumerate(ready, 1):
            entry = plan["benchmarks"][name]
            print(f"\nBenchmark {i}/{len(ready)}: {name}", flush=True)
            entry["status"] = "running"
            write_json(output / "plan.json", plan)
            try:
                if args.with_baseline:
                    run_benchmark(args, code_root, output, name, entry, "ar")
                run_benchmark(args, code_root, output, name, entry, "two_pass")
                entry["status"] = "completed"
            except Exception as exc:
                entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                print(f"Failed {name}: {exc}", flush=True)
            write_json(output / "plan.json", plan)
            write_report(output, plan)
    except KeyboardInterrupt:
        entry.update(status="interrupted", reason="Interrupted by user")
        write_json(output / "plan.json", plan)
        write_report(output, plan)
        print("Interrupted; completed results are preserved in", output, flush=True)
        return 130
    print("\n" + (output / "summary.md").read_text(), flush=True)
    print("Results:", output / "summary.csv", flush=True)
    return 1 if any(e["status"] in {"failed", "preparation_failed"} for e in plan["benchmarks"].values()) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
