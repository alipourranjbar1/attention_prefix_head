#!/usr/bin/env python3
"""Independent HTTP evaluation client for an already running vLLM server.

Usage: python evaluator.py --base-url http://127.0.0.1:8209 --output results/dflash
This file does not launch vLLM. No server script imports or executes this file.
Use the same prompt manifest for compared model endpoints.


The evaluator keeps ReTrace-compatible defaults but exposes benchmark and
vLLM generation controls on the command line. During evaluation it shows one progress
bar per dataset/configuration, prints one compact dataset summary when each dataset
finishes, and stores summary.csv plus compact run_info.json provenance.
The frozen prompt manifest is kept separately so different model runs can reuse the
exact same prompts. No generated benchmark program is executed and no task-accuracy
score is claimed.

Works with the editable run_server_dflash.sh and its target/selector graph paths.
For localhost, its current launch record is discovered automatically by port.
--server-manifest can select a record explicitly; other launchers still work.
The server controls prefix fast/graph mode; this HTTP client does not enable it.
Startup warmup defaults to 512 tokens per request and at least 32 reported
speculative verification rounds, outside all measured intervals. Match
--selector-warmup-steps to the server's DFLASH_PREFIX_VALIDATE_STEPS if changed.
With no server manifest, confirm in the server terminal that it prints:
PREFIX_PARITY_PASS ... mode=fast npu_graph=True; steady-state fast path enabled

cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/Evaluator

python evaluator.py \
  --base-url http://127.0.0.1:8216 \
  --model qwen3-4b-prefix-dflash \
  --datasets all \
  --manifest ./retrace_eval_prompts_all_v2.json \
  --num-prompts 2000 \
  --temperatures 0 \
  --concurrencies 1 \
  --top-p 1.0 \
  --top-k -1 \
  --max-new-tokens 2048 \
  --seed 42 \
  --prompt-seed 42 \
  --warmup-requests 1 \
  --selector-warmup-tokens 512 \
  --selector-warmup-steps 32 \
  --no-thinking \
  --output "results/prefix_dflash_ep6_bs16_layer1_full"


## --rebuild-manifest
"""

import argparse
import csv
import hashlib
import json
import math
import os

# This evaluator is an HTTP client and does not execute models locally.
# Prevent PyTorch from auto-loading torch_npu when transformers/tokenizers
# are imported. Otherwise evaluator-only environments may fail looking for
# CANN/HCCL libraries such as libhccl.so.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

DEFAULT_TARGET = (
    "/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/"
    "1cfa9a7208912126459214e8b04321603b3df60c"
)
COUNTS = {
    "gsm8k": 128,
    "math500": 128,
    "humaneval": 128,
    "mbpp": 128,
    "lcb": 128,
    "aime25": 30,
    "mtbench": 80,
    "alpaca": 128,
}
DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "default", "test"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test"),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test"),
    "aime25": ("MathArena/aime_2025", "default", "train"),
    "mtbench": ("HuggingFaceH4/mt_bench_prompts", "default", "train"),
    "alpaca": ("tatsu-lab/alpaca", "default", "train"),
}
SPEC_COUNTERS = {
    "vllm:spec_decode_num_drafts": "rounds",
    "vllm:spec_decode_num_draft_tokens": "drafted",
    "vllm:spec_decode_num_accepted_tokens": "accepted",
    "vllm:request_success": "finished",
    "vllm:generation_tokens": "generated",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temp, path)


def names_from(values):
    aliases = {
        "mt-bench": "mtbench",
        "livecodebench": "lcb",
        "aime2025": "aime25",
        "math-500": "math500",
    }
    names = [aliases.get(n.lower(), n.lower()) for s in values for n in s.split(",")]
    if names == ["all"]:
        return list(COUNTS)
    if len(set(names)) != len(names) or any(n not in COUNTS for n in names):
        raise ValueError(f"Choose distinct datasets from {list(COUNTS)}, or all")
    return names


def format_prompt(name, row):
    if name in ("gsm8k", "math500", "aime25"):
        question = (
            row.get("question")
            if name == "gsm8k"
            else row.get("problem", row.get("question"))
        )
        if not isinstance(question, str):
            raise ValueError(f"{name}: missing question/problem")
        return (
            question
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    if name == "humaneval":
        return (
            "Write a solution to the following problem and make sure that it passes the tests:\n```python\n"
            + row["prompt"]
            + "\n```"
        )
    if name == "mbpp":
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("mbpp: missing prompt")
        return prompt
    if name == "mtbench":
        return (row.get("turns") or row["prompt"])[0]
    if name == "alpaca":
        return row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
    if name == "lcb":
        starter = row.get("starter_code", "")
        return (
            row["question_content"]
            + "\n\n"
            + (
                "Complete the following Python starter code.\n```python\n"
                + starter
                + "\n```"
                if starter
                else "Write a Python program that reads from standard input and writes to standard output."
            )
            + "\nReturn your solution in a Python code block."
        )
    raise ValueError(name)


def load_rows(name, source, offline, lcb_release):
    if source:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            from datasets import DatasetDict, load_from_disk

            rows = load_from_disk(str(path))
            if isinstance(rows, DatasetDict):
                split = DATASETS.get(name, ("", "", "test"))[2]
                rows = rows[split]
            return rows, {"path": str(path), "fingerprint": rows._fingerprint}, False
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        else:
            rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(
                "Local source must be JSONL, a JSON row list, or datasets.save_to_disk directory"
            )
        return rows, {"path": str(path), "sha256": file_hash(path)}, True
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if name == "lcb":
        from huggingface_hub import hf_hub_download

        rows, hashes = [], {}
        for filename in [
            "test.jsonl",
            *[f"test{i}.jsonl" for i in range(2, lcb_release + 1)],
        ]:
            local = hf_hub_download(
                "livecodebench/code_generation_lite",
                filename,
                repo_type="dataset",
                local_files_only=offline,
            )
            hashes[filename] = file_hash(local)
            with open(local) as f:
                for line in f:
                    row = json.loads(line)
                    rows.append(
                        {
                            k: row.get(k, "")
                            for k in (
                                "question_content",
                                "starter_code",
                                "question_id",
                                "contest_date",
                            )
                        }
                    )
        return (
            rows,
            {
                "repo": "livecodebench/code_generation_lite",
                "release": lcb_release,
                "source_sha256": hashes,
            },
            False,
        )
    from datasets import load_dataset

    repo, config, split = DATASETS[name]
    # Do not perform a separate HfApi.dataset_info call: it breaks cached/offline runs.
    rows = load_dataset(repo, config, split=split)
    return (
        rows,
        {
            "repo": repo,
            "config": config,
            "split": split,
            "fingerprint": rows._fingerprint,
        },
        False,
    )


def token_list(value):
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized prompt")
        value = value[0]
    if (
        not isinstance(value, list)
        or not value
        or any(type(x) is not int or x < 0 for x in value)
    ):
        raise ValueError("Expected a nonempty list of nonnegative token IDs")
    return value


def prepare_manifest(args, names):
    """Freeze formatted token IDs once; reuse this exact file on all model endpoints."""
    if args.rebuild_manifest and args.manifest.exists():
        args.manifest.unlink()
    if args.manifest.exists():
        existing = json.loads(args.manifest.read_text())
        if existing.get("thinking") != args.thinking:
            raise ValueError(
                "Existing manifest uses a different thinking setting; "
                "use --rebuild-manifest or another --manifest"
            )
        missing = [n for n in names if n not in existing.get("datasets", {})]
        if missing:
            raise ValueError(
                f"Existing manifest lacks {missing}; use --rebuild-manifest "
                "or another --manifest"
            )
        return existing
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    sources = {}
    for item in args.source:
        name, sep, path = item.partition("=")
        if not sep or name not in COUNTS or name in sources:
            raise ValueError("--source must be a unique DATASET=LOCAL_PATH mapping")
        sources[name] = path
    result = {
        "format": 3,
        "seed": args.prompt_seed,
        "thinking": args.thinking,
        "target": str(args.target),
        "target_config_sha256": file_hash(args.target / "config.json"),
        "tokenizer_sha256": {
            p.name: file_hash(p)
            for p in sorted(args.target.iterdir())
            if p.name
            in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "special_tokens_map.json",
            )
        },
        "datasets": {},
        "author_row_ids_available": False,
        "choices": {
            "selection": "seeded shuffle; all rows when full set requested",
            "mtbench": "first user turn only; paper does not specify turn policy",
            "lcb_release": f"release_v{args.lcb_release}",
            "formatter_sha256": file_hash(__file__),
        },
    }
    for name in names:
        rows, source_info, local_json = load_rows(
            name, sources.get(name), args.offline, args.lcb_release
        )
        indices = list(range(len(rows)))
        requested_count = args.num_prompts if args.num_prompts > 0 else COUNTS[name]
        if len(rows) > requested_count:
            random.Random(args.prompt_seed).shuffle(indices)
        indices = indices[:requested_count]
        selected = []
        for i in indices:
            row = rows[i]
            if row.get("messages"):
                prompt = ""
            elif (
                local_json
                and isinstance(row.get("prompt"), str)
                and set(row) <= {"id", "prompt", "answer"}
            ):
                prompt = row["prompt"]
            else:
                prompt = row.get("user_prompt") or format_prompt(name, row)
            messages = row.get("messages") or [{"role": "user", "content": prompt}]
            tokens = token_list(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=args.thinking,
                )
            )
            selected.append(
                {
                    "source_row": i,
                    "source_id": str(
                        row.get("id", row.get("task_id", row.get("question_id", i)))
                    ),
                    "prompt_token_ids": tokens,
                    "sha256": digest(tokens),
                }
            )
        if not selected:
            raise ValueError(f"No rows for {name}")
        result["datasets"][name] = {
            "source": source_info,
            "source_rows": len(rows),
            "requests": len(selected),
            "results": selected,
        }
        print(f"Prepared {name}: {len(selected)} prompts", flush=True)
    result["complete_default_counts"] = all(
        len(result["datasets"].get(n, {}).get("results", [])) == c
        for n, c in COUNTS.items()
        if n in result["datasets"]
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing frozen set from another evaluator.
    with args.manifest.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return result


def select_prompts(manifest, names, limit, thinking):
    if manifest.get("thinking") is not thinking:
        raise ValueError(
            "Manifest thinking setting differs from the requested evaluation"
        )
    chosen = {}
    for name in names:
        if name not in manifest.get("datasets", {}):
            raise ValueError(
                f"Manifest lacks {name}; use another --manifest or select its datasets"
            )
        rows = manifest["datasets"][name]["results"]
        # Use up to the requested number of prompts.
        # If the dataset contains fewer examples, use the full available set.
        rows = rows[:limit] if limit else rows
        if not rows:
            raise ValueError(f"Empty prompt set: {name}")
        for row in rows:
            ids = token_list(row["prompt_token_ids"])
            if row.get("sha256") != digest(ids):
                raise ValueError(f"Manifest token hash mismatch in {name}")
        chosen[name] = rows
    return chosen


def normalize_url(value):
    p = urlsplit(value.rstrip("/"))
    if (
        p.scheme not in ("http", "https")
        or not p.netloc
        or p.query
        or p.fragment
        or p.username
    ):
        raise ValueError(
            "Use http(s)://host:port[/v1], without credentials or query parameters"
        )
    path = p.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((p.scheme, p.netloc, path, "", "")).rstrip("/")


def sampling(args, temperature, eos):
    """Build vLLM sampling parameters from explicit CLI controls."""
    top_p = args.top_p
    if top_p is None:
        top_p = 1.0 if temperature == 0 else 0.95

    top_k = args.top_k
    if top_k is None:
        top_k = 1 if temperature == 0 else 20

    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": args.min_p,
        "max_tokens": args.max_new_tokens,
        "seed": args.seed,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "repetition_penalty": args.repetition_penalty,
        "min_tokens": args.min_tokens,
        "ignore_eos": args.ignore_eos,
        "stop_token_ids": args.stop_token_ids if args.stop_token_ids is not None else eos,
        "skip_special_tokens": args.skip_special_tokens,
    }


def validate_spec(stats):
    def integer(key):
        value = stats.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid per-request speculative metric {key}: {value}")
        return value

    rounds, accepted, drafted, k = (
        integer(key)
        for key in (
            "num_spec_steps",
            "num_accepted_draft_tokens",
            "num_draft_tokens",
            "num_spec_tokens",
        )
    )
    if (
        accepted > drafted
        or drafted > rounds * k
        or (rounds and (k < 1 or drafted < rounds))
    ):
        raise ValueError(f"Inconsistent speculative counters: {stats}")
    hist = stats.get("acceptance_histogram")
    if (
        not isinstance(hist, list)
        or len(hist) != k + 1
        or any(type(v) is not int or v < 0 for v in hist)
        or sum(hist) != rounds
        or sum(i * n for i, n in enumerate(hist)) != accepted
    ):
        raise ValueError("Acceptance histogram disagrees with per-request counters")
    tau = 1 + accepted / rounds if rounds else None
    if rounds and not math.isclose(
        stats.get("mean_acceptance_length", -1), tau, abs_tol=1e-6
    ):
        raise ValueError("Server acceptance length disagrees with raw counts")
    return {
        "rounds": rounds,
        "accepted": accepted,
        "drafted": drafted,
        "k": k,
        "histogram": hist,
        "acceptance_length": tau,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


class Client:
    def __init__(self, base_url, model=None, timeout=3600, metrics_url=None):
        import requests

        self.session = requests.Session()
        self._thread_local = threading.local()
        self._api_key = os.getenv("OPENAI_API_KEY")
        if self._api_key:
            self.session.headers["Authorization"] = f"Bearer {self._api_key}"
        self.base = normalize_url(base_url)
        self._trust_env = urlsplit(self.base).hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
        if not self._trust_env:
            self.session.trust_env = False
        self.timeout = timeout
        response = self.session.get(self.base + "/v1/models", timeout=30)
        response.raise_for_status()
        models = response.json()["data"]
        if not model and len(models) != 1:
            raise ValueError(
                "Server exposes multiple models; select --model explicitly"
            )
        self.model = model or models[0]["id"]
        matching = [m for m in models if m["id"] == self.model]
        if len(matching) != 1:
            raise ValueError(f"Model {self.model} is not served at {self.base}")
        self.info = matching[0]
        self.metrics_url = metrics_url or self.base + "/metrics"

    def _request_session(self):
        """One persistent requests.Session per worker thread."""
        session = getattr(self._thread_local, "session", None)
        if session is None:
            import requests
            session = requests.Session()
            session.trust_env = self._trust_env
            if self._api_key:
                session.headers["Authorization"] = f"Bearer {self._api_key}"
            self._thread_local.session = session
        return session

    def complete(self, tokens, params):
        body = dict(
            params,
            model=self.model,
            prompt=tokens,
            n=1,
            stream=True,
            stream_options={"include_usage": True},
            return_token_ids=True,
            add_special_tokens=False,
        )
        started = time.perf_counter()
        first = last = None
        token_ids, texts, usage, metrics, finish, request_id = (
            [],
            [],
            None,
            None,
            None,
            None,
        )
        done = False
        session = self._request_session()
        with session.post(
            self.base + "/v1/completions",
            json=body,
            stream=True,
            timeout=(30, self.timeout),
        ) as response:
            if not response.ok:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:1500]}"
                )
            for raw in response.iter_lines(chunk_size=1):
                if not raw or not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(f"Server error: {event['error']}")
                request_id = event.get("id", request_id)
                if event.get("usage") is not None:
                    usage = event["usage"]
                if event.get("metrics") is not None:
                    metrics = event["metrics"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("Unexpected second completion stream")
                    delta = choice.get("token_ids")
                    if delta:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        token_ids.extend(delta)
                    texts.append(choice.get("text") or "")
                    finish = choice.get("finish_reason") or finish
        latency = time.perf_counter() - started
        if not done or usage is None or finish not in ("stop", "length"):
            raise RuntimeError(
                f"Incomplete generation: done={done}, usage={usage}, finish={finish}"
            )
        if usage.get("prompt_tokens") != len(tokens) or usage.get(
            "completion_tokens"
        ) != len(token_ids):
            raise ValueError(
                "API token IDs do not match usage; use the pinned vLLM return_token_ids support"
            )
        spec = (metrics or {}).get("speculative_decoding")
        return {
            "request_id": request_id,
            "output_token_ids": token_ids,
            "text": "".join(texts),
            "prompt_tokens": len(tokens),
            "completion_tokens": len(token_ids),
            "latency_seconds": latency,
            "ttft_seconds": first - started if first else None,
            "tpot_seconds": (last - first) / (len(token_ids) - 1)
            if first and len(token_ids) > 1
            else None,
            "finish_reason": finish,
            "engine_metrics": metrics,
            "speculation": validate_spec(spec) if spec is not None else None,
        }

    def snapshot(self):
        from prometheus_client.parser import text_string_to_metric_families

        response = self.session.get(self.metrics_url, timeout=30)
        response.raise_for_status()
        counters, positions = {}, {}
        active = 0.0
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if sample.labels.get("model_name") != self.model:
                    continue
                name = sample.name.removesuffix("_total")
                value = float(sample.value)
                if not math.isfinite(value):
                    continue
                if name in SPEC_COUNTERS:
                    key = SPEC_COUNTERS[name]
                    counters[key] = counters.get(key, 0.0) + value
                if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                    pos = sample.labels.get("position")
                    positions[pos] = positions.get(pos, 0.0) + value
                if name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
                    active += value
        if "finished" not in counters:
            raise ValueError(
                "Prometheus request counters missing for the served model; keep --disable-log-stats off"
            )
        return {
            "counters": counters,
            "positions": positions,
            "active": active,
            "raw": response.text,
        }

    def settle(self, before, expected, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            after = self.snapshot()
            difference = after["counters"]["finished"] - before["counters"]["finished"]
            if difference < 0:
                raise ValueError("Server counters reset during evaluation")
            if difference >= expected and after["active"] == 0:
                return after
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Server request metrics did not settle; no acceptance result will be fabricated"
                )
            time.sleep(0.2)


def counter_delta(before, after):
    result = {}
    for key in set(before["counters"]) | set(after["counters"]):
        if key not in before["counters"] or key not in after["counters"]:
            raise ValueError(
                f"Counter disappeared/appeared inside measured interval: {key}"
            )
        value = after["counters"][key] - before["counters"][key]
        if value < 0:
            raise ValueError(f"Counter reset: {key}")
        result[key] = value
    return result


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(rows, delta, wall_seconds, expect_speculative):
    specs = [r["speculation"] for r in rows]
    per_request = all(s is not None for s in specs)
    if not expect_speculative and any(s and s["rounds"] for s in specs):
        raise ValueError(
            "--target-only was specified but the server uses a draft model"
        )
    if per_request:
        counters = {
            k: sum(s[k] for s in specs) for k in ("rounds", "accepted", "drafted")
        }
        if len({s["k"] for s in specs}) != 1:
            raise ValueError("Proposal count changed during benchmark")
        source = "per_request_api"
        k = specs[0]["k"]
        hist = [sum(s["histogram"][i] for s in specs) for i in range(k + 1)]
        per_position = [
            sum(hist[i + 1 :]) / counters["rounds"] if counters["rounds"] else None
            for i in range(k)
        ]
        consistent = all(delta.get(key) == count for key, count in counters.items())
    elif expect_speculative:
        if (
            not all(k in delta for k in ("rounds", "accepted", "drafted"))
            or delta["rounds"] <= 0
        ):
            raise ValueError(
                "No speculative metrics: check draft/proposer launch; missing metrics are not zero acceptance"
            )
        counters = {k: delta[k] for k in ("rounds", "accepted", "drafted")}
        source, hist, per_position, consistent = "prometheus_interval", None, None, True
    else:
        if delta.get("rounds", 0) > 0:
            raise ValueError(
                "--target-only was specified but the server uses a draft model"
            )
        counters = {"rounds": 0, "accepted": 0, "drafted": 0}
        source, hist, per_position, consistent = "target_only", None, None, True
    d, a, p = (counters[k] for k in ("rounds", "accepted", "drafted"))
    if not 0 <= a <= p or (d > 0 and p < d):
        raise ValueError("Invalid accepted/drafted/round counters")
    tokens = sum(r["completion_tokens"] for r in rows)
    seconds = sum(r["latency_seconds"] for r in rows)
    exclusive = delta["finished"] == len(rows)
    if not exclusive and source == "prometheus_interval":
        raise ValueError(
            "Other requests contaminated aggregate acceptance metrics; use an idle server"
        )
    return {
        "requests": len(rows),
        "output_tokens": tokens,
        "wall_seconds": wall_seconds,
        "summed_request_seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "wall_tokens_per_second": tokens / wall_seconds,
        "request_tokens_per_second_mean": mean(
            [r["completion_tokens"] / r["latency_seconds"] for r in rows]
        ),
        "latency_seconds_mean": seconds / len(rows),
        "ttft_ms_mean": mean(
            [
                1000 * r["ttft_seconds"] if r["ttft_seconds"] is not None else None
                for r in rows
            ]
        ),
        "tpot_ms_request_mean": mean(
            [
                1000 * r["tpot_seconds"] if r["tpot_seconds"] is not None else None
                for r in rows
            ]
        ),
        "engine_ttft_ms_mean": mean(
            [(r["engine_metrics"] or {}).get("time_to_first_token_ms") for r in rows]
        ),
        "engine_mean_itl_ms_request_mean": mean(
            [(r["engine_metrics"] or {}).get("mean_itl_ms") for r in rows]
        ),
        "verification_rounds": d,
        "accepted_draft_tokens": a,
        "drafted_tokens": p,
        "accepted_per_round_pooled": a / d if d else None,
        "acceptance_length_with_bonus_pooled": 1 + a / d if d else None,
        "acceptance_length_with_bonus_request_mean": mean(
            [s["acceptance_length"] for s in specs]
        )
        if per_request
        else None,
        "draft_acceptance_rate": a / p if p else None,
        "acceptance_histogram": hist,
        "per_position_acceptance_rate": per_position,
        "acceptance_source": source,
        "counters_crosscheck": consistent,
        "exclusive_server_interval": exclusive,
        "measurement_valid": exclusive and consistent,
        "length_limited_requests": sum(r["finish_reason"] == "length" for r in rows),
        "greedy_parity": None,
        "comparison_valid": None,
        "speedup_tps": None,
        "speedup_total_latency": None,
        "speedup_request_tps_mean": None,
    }


def compare(cell, reference, reference_report, settings):
    issues = []
    if not reference_report.get("complete"):
        issues.append("reference run incomplete")
    if settings != reference_report.get("comparison_settings"):
        issues.append("target/tokenizer/generation settings differ")
    key = (cell["benchmark"], cell["temperature"], cell.get("concurrency", 1))
    old = next(
        (
            c
            for c in reference
            if (
                c["benchmark"],
                c["temperature"],
                c.get("concurrency", 1),
            )
            == key
        ),
        None,
    )
    if old is None:
        issues.append("reference cell missing")
    elif "results" not in old:
        issues.append("reference has no per-request outputs; output parity cannot be checked")
    elif [r["prompt_sha256"] for r in old["results"]] != [
        r["prompt_sha256"] for r in cell["results"]
    ]:
        issues.append("ordered prompts differ")
    mismatches = []
    if not issues and cell["temperature"] == 0:
        for i, (a, b) in enumerate(zip(old["results"], cell["results"])):
            if a["output_token_ids"] != b["output_token_ids"]:
                mismatches.append(i)
        cell["summary"]["greedy_parity"] = not mismatches
        if mismatches:
            issues.append("greedy output token IDs differ")
    if not cell["summary"]["measurement_valid"] or (
        old and not old["summary"]["measurement_valid"]
    ):
        issues.append("measurement not isolated or counters disagree")
    summary = cell["summary"]
    summary["comparison_valid"] = not issues
    summary["comparison_issues"] = issues
    summary["mismatching_requests"] = mismatches
    summary["reference_method"] = reference_report.get("method")
    summary["comparison_provenance_verified"] = bool(
        settings.get("target_weight_identity") and settings.get("runtime_identity")
    )
    if not issues:
        base = old["summary"]
        summary["speedup_tps"] = (
            summary["tokens_per_second"] / base["tokens_per_second"]
        )
        summary["speedup_total_latency"] = (
            base["summed_request_seconds"] / summary["summed_request_seconds"]
        )
        summary["speedup_request_tps_mean"] = (
            summary["request_tokens_per_second_mean"]
            / base["request_tokens_per_second_mean"]
        )


SUMMARY_FIELDS = [
    "method",
    "temperature",
    "benchmark",
    "requests",
    "tokens_per_second",
    "acceptance_length_with_bonus_pooled",
    "acceptance_length_with_bonus_request_mean",
    "draft_acceptance_rate",
    "ttft_ms_mean",
    "tpot_ms_request_mean",
]


def summary_row(report, cell):
    return {
        "method": report["method"],
        "temperature": cell["temperature"],
        "benchmark": cell["benchmark"],
        **{key: cell["summary"].get(key) for key in SUMMARY_FIELDS[3:]},
    }


def write_summary_csv(output, report):
    """Keep the original CSV columns and also retain compact run provenance."""
    path = output / "summary.csv"
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for cell in report["cells"]:
            writer.writerow(summary_row(report, cell))
    os.replace(temp, path)
    write_json(output / "run_info.json", report)


def _display_value(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_dataset_summary(report, cell):
    row = summary_row(report, cell)
    print(f"\n--- Final report: {cell['benchmark']} ---", flush=True)
    for key in SUMMARY_FIELDS:
        print(f"{key}: {_display_value(row.get(key))}", flush=True)
    print(flush=True)


def print_final_summary(report):
    """Print only the requested CSV columns at the end of the full evaluation."""
    print("\n================ FINAL EVALUATION SUMMARY ================", flush=True)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
    writer.writeheader()
    for cell in report["cells"]:
        writer.writerow(summary_row(report, cell))
    print("==========================================================", flush=True)


def clean_output_artifacts(output):
    """Remove files produced by older verbose evaluator versions from this folder."""
    exact = {
        "summary.csv",
        "report.json",
        "macro_summary.json",
        "prompts_used.json",
        "run_info.json",
    }
    patterns = (
        "*_before.prom",
        "*_after.prom",
        "*_requests.jsonl",
    )
    for name in exact:
        path = output / name
        if path.is_file():
            path.unlink()
    for pattern in patterns:
        for path in output.glob(pattern):
            if path.is_file():
                path.unlink()


class ProgressBar:
    """tqdm-backed progress bar with a dependency-free terminal fallback."""

    def __init__(self, total, label):
        self.total = total
        self.count = 0
        self.label = label
        self._bar = (
            tqdm(
                total=total,
                desc=label,
                unit="req",
                dynamic_ncols=True,
                leave=True,
            )
            if tqdm is not None
            else None
        )
        if self._bar is None:
            self._render()

    def _render(self):
        width = 30
        ratio = self.count / self.total if self.total else 1.0
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{self.label}: [{bar}] {self.count}/{self.total}",
            end="",
            flush=True,
        )

    def update(self, n=1):
        self.count += n
        if self._bar is not None:
            self._bar.update(n)
        else:
            self._render()

    def close(self):
        if self._bar is not None:
            self._bar.close()
        else:
            print(flush=True)

def client_runtime():
    """Select this Conda environment's C++ ABI before optional data-loader imports."""
    lib = Path(sys.prefix) / "lib/libstdc++.so.6"
    if os.getenv("RETRACE_EVAL_CPP_READY") or not lib.is_file():
        return
    if b"CXXABI_1.3.15\0" not in lib.read_bytes():
        return
    selected = [
        str(p.resolve()) for p in [lib.parent / "libgcc_s.so.1", lib] if p.is_file()
    ]
    retained = [
        p
        for p in re.split(r"[:\s]+", os.getenv("LD_PRELOAD", ""))
        if p and not Path(p).name.startswith(("libstdc++.so", "libgcc_s.so"))
    ]
    env = dict(
        os.environ, LD_PRELOAD=":".join(selected + retained), RETRACE_EVAL_CPP_READY="1"
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def parse_concurrencies(value):
    values = []
    for piece in str(value).split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            v = int(piece)
        except ValueError as exc:
            raise ValueError(
                f"Invalid concurrency {piece!r}; use e.g. 1 or 1,4,8"
            ) from exc
        if v <= 0:
            raise ValueError("Concurrency values must be positive integers")
        values.append(v)
    if not values:
        raise ValueError("At least one concurrency value is required")
    if len(set(values)) != len(values):
        raise ValueError("Concurrency values must be distinct")
    return values


def run_request_batch(client, prompts, params, concurrency, label):
    rows = [None] * len(prompts)
    progress = ProgressBar(len(prompts), label)

    try:
        if concurrency == 1:
            for i, prompt in enumerate(prompts):
                row = client.complete(prompt["prompt_token_ids"], params)
                row.update(
                    prompt_sha256=prompt["sha256"],
                    source_row=prompt.get("source_row"),
                    source_id=prompt.get("source_id"),
                )
                rows[i] = row
                progress.update()
            return rows

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(client.complete, prompt["prompt_token_ids"], params): (i, prompt)
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                i, prompt = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"{label}: request {i + 1}/{len(prompts)} failed"
                    ) from exc
                row.update(
                    prompt_sha256=prompt["sha256"],
                    source_row=prompt.get("source_row"),
                    source_id=prompt.get("source_id"),
                )
                rows[i] = row
                progress.update()
    finally:
        progress.close()

    if any(row is None for row in rows):
        raise RuntimeError(f"{label}: missing one or more request results")
    return rows

def run_warmups(client, prompts, params, concurrency, count):
    if count <= 0:
        return
    warmup_prompts = [prompts[i % len(prompts)] for i in range(count)]
    if concurrency == 1:
        for prompt in warmup_prompts:
            client.complete(prompt["prompt_token_ids"], params)
        return

    with ThreadPoolExecutor(max_workers=min(concurrency, count)) as pool:
        futures = [
            pool.submit(client.complete, p["prompt_token_ids"], params)
            for p in warmup_prompts
        ]
        for future in as_completed(futures):
            future.result()


INTEGRATED_SERVER_FORMATS = {
    "dflash_evaluator_integration_v1", "dflash_evaluator_integration_v2",
    "dflash_evaluator_integration_v3",
}


def is_integrated_server(manifest):
    return bool(manifest and manifest.get("format") in INTEGRATED_SERVER_FORMATS)


def check_server_process(manifest):
    try:
        pid = int(manifest["launcher_pid"])
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        valid = fields[19] == str(manifest["launcher_start_ticks"]) and fields[0] != "Z"
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise ValueError("Launch manifest is stale: its server process is not running") from exc
    if not valid:
        raise ValueError("Launch manifest is stale: server process identity changed")


def read_server_manifest(path, client, target_config_hash):
    """Read this launch's provenance; never use an old log to certify a new server."""
    actual = urlsplit(client.base)
    loopback = {"localhost", "127.0.0.1", "::1"}
    discovered = path is None
    if discovered:
        if actual.hostname not in loopback:
            return None
        root = Path(__file__).resolve().parent.parent
        path = root / "output/evaluator_servers" / f"port_{actual.port or 80}.json"
        if not path.is_file():
            return None
    manifest = json.loads(path.read_text())
    if discovered:
        try:
            check_server_process(manifest)
        except ValueError as exc:
            print(f"Ignoring old launch record {path}: {exc}. Current server mode is unverified.", flush=True)
            return None
    if manifest.get("served_model", manifest.get("served_model_name")) != client.model:
        raise ValueError("Server provenance file belongs to a different served model")
    if not is_integrated_server(manifest):
        return manifest
    if manifest.get("check_only"):
        raise ValueError("A CHECK_ONLY launch record does not identify a running server")
    claimed = urlsplit(normalize_url(manifest["base_url"]))
    same_host = (claimed.hostname == actual.hostname or
                 claimed.hostname in loopback and actual.hostname in loopback)
    if not same_host or (claimed.port or 80) != (actual.port or 80) or claimed.scheme != actual.scheme:
        raise ValueError("--server-manifest belongs to another server URL/port")
    if manifest.get("target_config_sha256") != target_config_hash:
        raise ValueError("Launch manifest and evaluator target configurations differ")
    if actual.hostname in loopback:
        check_server_process(manifest)
    manifest = dict(manifest, manifest_path=str(path.resolve()), automatically_discovered=discovered)
    print(f"Server variant: {manifest['variant']}; draft block: {manifest['block_size']}", flush=True)
    print(f"Target: {manifest['target']}\nDraft:  {manifest['draft']}", flush=True)
    print(f"Prefix mode={manifest.get('prefix_mode')}; graph={manifest['prefix_npu_graph']}; "
          f"token_cache={manifest['prefix_token_cache']}", flush=True)
    if "target_execution_requested" in manifest:
        print(f"Target execution requested: {manifest['target_execution_requested']}; "
              f"draft execution requested: {manifest.get('draft_execution_requested', 'eager')}; "
              "full target verification retained.", flush=True)
        if manifest.get("draft_backend_note"):
            print(manifest["draft_backend_note"], flush=True)
    return manifest


def run_selector_warmup(client, prompt, params, count, required_steps=32, expected_k=None):
    """Untimed sequential requests until the requested warmup coverage is reached.

    HTTP counters confirm verification rounds, not that graph mode is enabled.
    The server's parity log remains the evidence for its actual execution mode.
    """
    if count == 0:
        return {"enabled": False, "required_verification_rounds": required_steps,
                "excluded_from_measurement": True}
    warmup = dict(params, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0,
                  max_tokens=count, min_tokens=count, ignore_eos=True, stop_token_ids=[])
    requests, tokens, rounds, observed_k = 0, 0, 0, None
    sources = set()
    print(f"Startup warmup: {count} tokens/request, at least {required_steps} verification rounds; "
          "excluded from all reported timing and acceptance counters.", flush=True)
    while requests == 0 or rounds < required_steps:
        before = client.snapshot()
        if before["active"]:
            raise ValueError("Server is busy before selector warmup")
        row = client.complete(prompt["prompt_token_ids"], warmup)
        if row["completion_tokens"] != count:
            raise ValueError(f"Startup warmup returned {row['completion_tokens']} tokens; expected {count}")
        spec = row.get("speculation")
        if expected_k is not None and (not spec or spec["k"] != expected_k):
            raise ValueError("Warmup proposal count does not match the selected launch manifest")
        if spec is not None:
            if observed_k is not None and spec["k"] != observed_k:
                raise ValueError("Proposal count changed during selector warmup")
            observed_k = spec["k"]
        after = client.settle(before, 1)
        delta = counter_delta(before, after)
        if delta["finished"] != 1:
            raise ValueError("Other requests overlapped the selector warmup")
        if spec is not None:
            reached = spec["rounds"]
            sources.add("per_request_api")
        else:
            reached = delta.get("rounds", 0)
            sources.add("prometheus_interval" if "rounds" in delta else "unavailable")
        if not math.isfinite(reached) or reached < 0 or reached != int(reached):
            raise ValueError("Invalid warmup verification-round count")
        requests += 1
        tokens += row["completion_tokens"]
        rounds += int(reached)
        if required_steps and reached == 0:
            raise ValueError("Warmup produced no speculative verification rounds. Check the draft server, "
                             "increase --selector-warmup-tokens, or use --target-only for a target-only server.")
        print(f"  Warmup {requests}: {rounds} verification rounds accumulated; "
              f"draft proposals={observed_k if observed_k is not None else 'unavailable'}", flush=True)
    return {"enabled": True, "requests": requests, "completion_tokens": tokens,
            "verification_rounds": rounds, "num_speculative_tokens": observed_k,
            "required_verification_rounds": required_steps,
            "counter_sources": sorted(sources),
            "excluded_from_measurement": True}


def confirm_prefix_runtime(manifest):
    """Confirm the current launch's parity marker outside the measured interval."""
    if not is_integrated_server(manifest):
        return {"verified": False, "reason": "No integrated launch manifest provided"}
    if manifest["variant"] != "prefix" or manifest["prefix_mode"] == "reference":
        return {"verified": False, "reason": "No optimized prefix runtime requested"}
    mode = manifest["prefix_mode"]
    graph = manifest["prefix_npu_graph"]
    needed = manifest["prefix_validate_steps"]
    log = Path(manifest["server_log"])
    compiled = manifest.get("kind") == "compiled_prefix_v3"
    marker = (r"\bCOMPILED_PREFIX_PARITY_PASS kind=compiled_prefix_v3 " if compiled
              else r"(?<!COMPILED_)\bPREFIX_PARITY_PASS ")
    pattern = re.compile(marker + r"checked_blocks=(\d+)[^\n]*"
                         + rf"mode={mode} npu_graph={graph}")
    # tee may still be draining the final warmup log line. This is untimed.
    deadline = time.monotonic() + 5
    while True:
        matches = []
        if log.is_file():
            with log.open(errors="replace") as stream:
                for line in stream:
                    if re.search(r"\b(?:COMPILED_)?PREFIX_PARITY_FAIL\b", line):
                        raise ValueError(f"Selector parity failed in {log}; no fast result can be certified")
                    match = pattern.search(line)
                    if match and (mode != "fast" or "steady-state fast path enabled" in line):
                        if compiled and "production_tokens_checked=True" not in line:
                            continue
                        matches.append(int(match.group(1)))
        ready = matches and max(matches) >= needed
        if ready:
            if mode == "validate":
                print("VALIDATION MODE: every block executes reference comparisons; throughput is diagnostic.", flush=True)
            else:
                print(f"Prefix ready: mode=fast, graph={graph}, initial parity checks completed.", flush=True)
            return {"verified": True, "mode": mode, "npu_graph": graph,
                    "kind": manifest.get("kind", "local_prefix_v2"),
                    "checked_blocks_at_warmup": max(matches), "server_log": str(log),
                    "scope": "Startup selector parity on observed blocks; not universal output parity"}
        if time.monotonic() >= deadline:
            raise ValueError(f"Expected prefix parity marker missing from {log}; inspect the server before timing")
        time.sleep(0.1)


def confirm_dspark_runtime(manifest):
    if not is_integrated_server(manifest) or manifest.get("dspark_head_graph", "off") == "off":
        return {"verified": False, "reason": "No DSpark graph requested"}
    mode = manifest["dspark_head_graph"]
    log = Path(manifest["server_log"])
    deadline = time.monotonic() + 5
    while True:
        text = log.read_text(errors="replace") if log.is_file() else ""
        if "DSPARK_HEAD_PARITY_FAIL" in text:
            raise ValueError(f"DSpark parity failure: {log}")
        pattern = rf"DSPARK_HEAD_PARITY_PASS checked_blocks=(\d+) max_score_error=0 mode={mode} npu_graph=True production_tokens_checked=True"
        counts = [int(m.group(1)) for m in re.finditer(pattern, text)]
        if counts and max(counts) >= manifest["dspark_validate_steps"]:
            if mode == "fast" and "steady-state fast path enabled" not in text:
                raise ValueError("DSpark steady-state fast path not confirmed")
            return {"verified": True, "mode": mode, "checked_blocks_at_warmup": max(counts),
                    "scope": "Observed startup blocks only; not universal parity"}
        if time.monotonic() >= deadline:
            raise ValueError(f"DSpark graph parity marker missing: {log}")
        time.sleep(.1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--version", action="version",
                   version="DFlash HTTP evaluator 2026-10-13.1 (integrated target/selector graphs)")
    p.add_argument("--base-url", help="http://host:port or http://host:port/v1")
    p.add_argument("--model", help="Automatically discovered when one model is served")
    p.add_argument("--method", help="Result label; defaults to launch manifest's variant label, then served model name")
    p.add_argument(
        "--target",
        type=Path,
        default=Path(os.getenv("RETRACE_TARGET", DEFAULT_TARGET)),
    )
    p.add_argument("--metrics-url")

    p.add_argument(
        "--datasets",
        "--dataset",
        nargs="+",
        default=["all"],
        help="Available: " + ", ".join(COUNTS) + ", or all",
    )
    p.add_argument(
        "--num-prompts",
        "--limit",
        dest="num_prompts",
        type=int,
        default=0,
        help="Requests per selected dataset; 0 uses the dataset default",
    )
    p.add_argument("--source", action="append", default=[])
    p.add_argument("--offline", action="store_true")
    p.add_argument("--lcb-release", type=int, default=5)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().with_name("retrace_eval_prompts.json"),
    )
    p.add_argument("--rebuild-manifest", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--prompt-seed", type=int, default=42)
    p.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    p.add_argument(
        "--concurrencies",
        "--concurrency",
        default="1",
        help="One value or comma-separated sweep, e.g. 1 or 1,4,8,32",
    )

    p.add_argument(
        "--temperatures",
        "--temperature",
        nargs="+",
        type=float,
        default=[0.0, 1.0],
    )
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--min-tokens", type=int, default=0)
    p.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--skip-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--stop-token-ids", nargs="+", type=int, default=None)

    p.add_argument(
        "--max-new-tokens",
        "--max-tokens",
        "--sequence-length",
        dest="max_new_tokens",
        type=int,
        default=8192,
        help="Maximum generated/output tokens per request",
    )
    p.add_argument(
        "--max-sequence-length",
        type=int,
        default=0,
        help="Total prompt+generation safety cap; 0 uses server max_model_len",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--selector-warmup-tokens", type=int, default=512,
                   help="Tokens per untimed startup request; requests repeat until --selector-warmup-steps is reached; 0 disables")
    p.add_argument("--selector-warmup-steps", type=int, default=32,
                   help="Minimum reported verification rounds during startup; match the server's initial selector checks; 0 disables the minimum")
    p.add_argument("--timeout-s", type=float, default=3600)
    p.add_argument("--target-only", action="store_true")
    p.add_argument("--reference", type=Path)
    p.add_argument("--server-manifest", type=Path,
                   help="Current launch.json; automatically discovered by localhost port for run_server_dflash.sh")
    p.add_argument("--output", type=Path, help="Output directory for summary.csv and compact run_info.json")
    p.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove artifacts from older verbose evaluator runs in --output before starting",
    )

    args = p.parse_args()
    args.prompt_seed = args.seed if args.prompt_seed is None else args.prompt_seed

    try:
        concurrencies = parse_concurrencies(args.concurrencies)
    except ValueError as exc:
        p.error(str(exc))

    if args.num_prompts < 0:
        p.error("--num-prompts must be >= 0")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be >= 1")
    if args.max_sequence_length < 0:
        p.error("--max-sequence-length must be >= 0")
    if args.timeout_s <= 0:
        p.error("--timeout-s must be > 0")
    if args.warmup_requests < 0:
        p.error("--warmup-requests must be >= 0")
    if args.selector_warmup_tokens < 0:
        p.error("--selector-warmup-tokens must be >= 0")
    if args.selector_warmup_steps < 0:
        p.error("--selector-warmup-steps must be >= 0")
    if not 1 <= args.lcb_release <= 10:
        p.error("--lcb-release must be in [1, 10]")
    if len(set(args.temperatures)) != len(args.temperatures):
        p.error("--temperatures must not contain duplicates")
    if any(t < 0 for t in args.temperatures):
        p.error("temperatures must be >= 0")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        p.error("--top-p must be in (0, 1]")
    if args.top_k is not None and args.top_k != -1 and args.top_k < 1:
        p.error("--top-k must be -1 (disabled) or >= 1")
    if not 0 <= args.min_p <= 1:
        p.error("--min-p must be in [0, 1]")
    if args.repetition_penalty <= 0:
        p.error("--repetition-penalty must be > 0")
    if args.min_tokens < 0:
        p.error("--min-tokens must be >= 0")
    if args.stop_token_ids is not None and any(x < 0 for x in args.stop_token_ids):
        p.error("--stop-token-ids must be nonnegative")
    if not args.prepare_only and not args.base_url:
        p.error("--base-url is required unless --prepare-only")

    client_runtime()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    names = names_from(args.datasets)
    manifest = prepare_manifest(args, names)
    selected = select_prompts(
        manifest,
        names,
        args.num_prompts,
        args.thinking,
    )

    target_config_hash = file_hash(args.target / "config.json")
    if manifest.get("target_config_sha256") != target_config_hash:
        raise ValueError("Manifest was prepared with a different target configuration")
    for name, expected in manifest.get("tokenizer_sha256", {}).items():
        if Path(name).name != name or file_hash(args.target / name) != expected:
            raise ValueError("Manifest tokenizer files differ from the local target")

    if args.prepare_only:
        print(f"Frozen prompts: {args.manifest.resolve()}")
        return

    client = Client(args.base_url, args.model, args.timeout_s, args.metrics_url)

    eos = json.loads(
        (args.target / "generation_config.json").read_text()
    ).get("eos_token_id", 151645)
    eos = [eos] if isinstance(eos, int) else eos
    if not eos or any(type(t) is not int or t < 0 for t in eos):
        raise ValueError("Invalid target EOS configuration")

    server_max_len = client.info.get("max_model_len")
    effective_max_len = args.max_sequence_length or server_max_len

    if (
        args.max_sequence_length
        and server_max_len
        and args.max_sequence_length > server_max_len
    ):
        raise ValueError(
            f"--max-sequence-length={args.max_sequence_length} exceeds "
            f"server max_model_len={server_max_len}; change the server launcher first"
        )

    if effective_max_len and any(
        len(r["prompt_token_ids"]) + args.max_new_tokens + 16 > effective_max_len
        for rows in selected.values()
        for r in rows
    ):
        raise ValueError(
            "A full prompt + max_new_tokens + draft safety buffer exceeds "
            f"the effective sequence limit ({effective_max_len})."
        )

    server_manifest = read_server_manifest(args.server_manifest, client, target_config_hash)
    if args.target_only and is_integrated_server(server_manifest):
        raise ValueError("The supplied server manifest declares a speculative draft; remove --target-only")
    if (is_integrated_server(server_manifest) and server_manifest.get("prefix_npu_graph")
            and any(c != 1 for c in concurrencies)):
        raise ValueError("The integrated selector graph is measured at concurrency=1; use --concurrencies 1")
    method = args.method or ((server_manifest or {}).get("suggested_method")) or ("ar" if args.target_only else client.model)
    output = args.output or Path(
        f"eval_{method}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clean_output_artifacts(output)
    elif any((output / name).exists() for name in ("summary.csv", "run_info.json", "report.json")):
        raise ValueError("Output already contains results; choose a fresh --output or explicitly use --clean-output")

    resolved_sampling = {
        f"{t:g}": sampling(args, t, eos)
        for t in args.temperatures
    }

    settings = {
        "target_config_sha256": target_config_hash,
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "generation_seed": args.seed,
        "prompt_seed": args.prompt_seed,
        "max_new_tokens": args.max_new_tokens,
        "max_sequence_length": effective_max_len,
        "eos_token_ids": eos,
        "thinking": args.thinking,
        "concurrencies": concurrencies,
        "temperatures": args.temperatures,
        "resolved_sampling": resolved_sampling,
        "sampling_protocol": "manual-vllm",
        "stream": True,
        "n": 1,
        "target_weight_identity": (server_manifest or {}).get("target_sha256"),
        "runtime_identity": (
            {
                k: server_manifest.get(k)
                for k in (
                    "device",
                    "tp_size",
                    "dtype",
                    "max_model_len",
                    "prefix_caching",
                    "enforce_eager",
                    "model_runner",
                    "commits",
                    "target_execution_requested",
                    "speculative_method",
                    "draft_execution_requested",
                    "draft_enforce_eager",
                    "compilation_config",
                    "additional_config",
                )
            }
            if server_manifest
            else None
        ),
    }

    reference = json.loads(args.reference.read_text()) if args.reference else None

    report = {
        "format": 4,
        "complete": False,
        "method": method,
        "base_url": client.base,
        "model": client.model,
        "model_info": client.info,
        "manifest_sha256": file_hash(args.manifest),
        "comparison_settings": settings,
        "temperatures": args.temperatures,
        "concurrencies": concurrencies,
        "datasets": names,
        "server_manifest": server_manifest,
        "eval_code_sha256": file_hash(__file__),
        "argv": sys.argv,
        "paper": "https://arxiv.org/pdf/2608.29748",
        "cells": [],
        "limits": [
            "Stored-response training and Ascend vLLM are adaptations, not an exact paper-table reproduction.",
            "Acceptance length=1+accepted draft tokens/verification rounds (bonus convention).",
            "MT-Bench uses the frozen first-turn policy.",
            "Streaming is retained so TTFT and per-request speculative metrics remain available.",
            "Generated benchmark programs are never executed by this evaluator.",
        ],
    }

    warmup_prompt = next(iter(selected.values()))[0]
    warmup_tokens = args.selector_warmup_tokens
    required_steps = 0 if args.target_only else args.selector_warmup_steps
    expected_k = None
    integrated = is_integrated_server(server_manifest)
    if integrated:
        required_steps = max(required_steps, server_manifest["prefix_validate_steps"], server_manifest.get("dspark_validate_steps", 0))
        expected_k = server_manifest["num_speculative_tokens"]
        if warmup_tokens:
            warmup_tokens = max(warmup_tokens, server_manifest["block_size"] * required_steps)
    if effective_max_len and len(warmup_prompt["prompt_token_ids"]) + warmup_tokens + 16 > effective_max_len:
        raise ValueError("Startup warmup exceeds the server sequence limit; lower --selector-warmup-tokens or use a shorter prompt")
    report["performance_measurement"] = not (
        integrated and (server_manifest.get("prefix_mode") == "validate" or server_manifest.get("dspark_head_graph") == "validate" or server_manifest.get("graph_metrics")))
    report["startup_warmup"] = {"complete": False}

    try:
        report["startup_warmup"] = run_selector_warmup(
            client, warmup_prompt, sampling(args, 0.0, eos), warmup_tokens,
            required_steps=required_steps, expected_k=expected_k,
        )
        report["startup_warmup"]["complete"] = True
        report["prefix_runtime_confirmation"] = confirm_prefix_runtime(server_manifest)
        report["dspark_runtime_confirmation"] = confirm_dspark_runtime(server_manifest)
        if integrated and server_manifest["variant"] in {"vanilla", "dspark"}:
            family = "DSpark" if server_manifest["variant"] == "dspark" else "Vanilla DFlash"
            print(f"{family}: no prefix selector checks apply. Target graph settings are recorded.", flush=True)
        elif not report["prefix_runtime_confirmation"]["verified"] and not args.target_only:
            status = "Startup warmup completed." if warmup_tokens else "Startup warmup disabled."
            print(f"{status} The HTTP counters do not identify prefix graph mode. "
                  "Check PREFIX_PARITY_PASS or COMPILED_PREFIX_PARITY_PASS in the server terminal; also confirm "
                  "steady-state fast path enabled.", flush=True)
        write_summary_csv(output, report)
        for concurrency in concurrencies:
            for temperature in args.temperatures:
                params = sampling(args, temperature, eos)

                for name, prompts in selected.items():
                    print(
                        f"\nEvaluating {name} "
                        f"(T={temperature:g}, concurrency={concurrency})",
                        flush=True,
                    )

                    before_warmup = client.snapshot()
                    if before_warmup["active"]:
                        raise ValueError(
                            "Server is busy; evaluate each method in isolation"
                        )

                    run_warmups(
                        client,
                        prompts,
                        params,
                        concurrency,
                        args.warmup_requests,
                    )
                    if args.warmup_requests:
                        client.settle(before_warmup, args.warmup_requests)

                    before = client.snapshot()
                    if before["active"]:
                        raise ValueError("Server still has active requests after warmup")

                    started = time.perf_counter()
                    rows = run_request_batch(
                        client,
                        prompts,
                        params,
                        concurrency,
                        label=f"{name} T={temperature:g} c={concurrency}",
                    )
                    wall_seconds = time.perf_counter() - started

                    after = client.settle(before, len(prompts))
                    delta = counter_delta(before, after)
                    cell = {
                        "benchmark": name,
                        "temperature": temperature,
                        "concurrency": concurrency,
                        "sampling": params,
                        "prometheus_delta": delta,
                        "results": rows,
                        "summary": aggregate(
                            rows,
                            delta,
                            wall_seconds,
                            not args.target_only,
                        ),
                    }

                    if reference is not None:
                        compare(
                            cell,
                            reference["cells"],
                            reference,
                            settings,
                        )

                    # Retain compact, ordered output identities for cross-method
                    # greedy parity; never confuse acceptance equality with output equality.
                    cell["speculative_hashes"] = [hashlib.sha256(json.dumps(
                        {key: (r.get("speculation") or {}).get(key) for key in
                         ("rounds", "accepted", "drafted", "k", "histogram")},
                        sort_keys=True, separators=(",", ":")
                    ).encode()).hexdigest() for r in rows]
                    cell["prompt_hashes"] = [r["prompt_sha256"] for r in rows]
                    cell["output_hashes"] = [hashlib.sha256(json.dumps(
                        [r["output_token_ids"], r["finish_reason"]], separators=(",", ":")
                    ).encode()).hexdigest() for r in rows]
                    # Per-request rows are needed only transiently for aggregation/reference checks.
                    # Do not retain or write them after this dataset is summarized.
                    cell.pop("results", None)
                    cell.pop("prometheus_delta", None)
                    report["cells"].append(cell)
                    write_summary_csv(output, report)
                    print_dataset_summary(report, cell)

        report["complete"] = True
        write_summary_csv(output, report)

    except BaseException:
        # Preserve summaries for all datasets that completed before the failure.
        write_summary_csv(output, report)
        raise

    print_final_summary(report)
    print(f"Results: {output.resolve()}/summary.csv", flush=True)


if __name__ == "__main__":
    main()

