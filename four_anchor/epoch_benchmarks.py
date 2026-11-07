"""Durable epoch-checkpoint queue; one separate evaluator NPU, no rank barriers.

Only a complete checkpoint published by training rank zero becomes a job.
One evaluator drains jobs serially while later training epochs continue.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_epoch(directory, checkpoint, epoch, step):
    checkpoint = Path(checkpoint).resolve()
    # save() calls this after full model/optimizer export and all RNG writes.
    checksums = (checkpoint / "checkpoint_sha256.txt").read_text()
    job = {"epoch": epoch, "step": step, "checkpoint": str(checkpoint),
           "checkpoint_sha256": checksums}
    path = Path(directory) / "queue" / f"epoch-{epoch:03d}.json"
    if path.exists() and json.loads(path.read_text()) != job:
        raise ValueError(f"Epoch {epoch} already refers to a different checkpoint; use a new run directory")
    write_json(path, job)
    print(json.dumps({"event": "epoch_benchmark_queued", **job}), flush=True)


def validate_checkpoint(job):
    checkpoint = Path(job["checkpoint"])
    expected = dict(line.split(maxsplit=1)[::-1] for line in job["checkpoint_sha256"].splitlines())
    if set(expected) != {"model.safetensors", "four_anchor_config.json"}:
        raise ValueError("Incomplete checkpoint checksums")
    for name, value in expected.items():
        if digest(checkpoint / name) != value:
            raise ValueError(f"Checkpoint changed after epoch publication: {name}")


def report(directory):
    directory = Path(directory)
    rows, states = [], []
    for path in sorted((directory / "states").glob("epoch-*.json")):
        state = json.loads(path.read_text())
        states.append(state)
        summary = Path(state["output"]) / "suite_summary.json"
        entries = json.loads(summary.read_text())["benchmarks"] if summary.exists() else {
            "suite": {"status": state["status"], "reason": state.get("reason", "")}}
        for name, entry in entries.items():
            s = entry.get("summary", {})
            rows.append({"epoch": state["epoch"], "step": state["step"], "benchmark": name,
                "status": entry["status"], "prompts": s.get("prompts", entry.get("prompts", 0)),
                "acceptance_including_bonus": s.get("acceptance_length_including_bonus"),
                "tokens_per_second": s.get("tokens_per_second_including_prefill"),
                "extra_target_forwards": s.get("extra_target_forwards"),
                "checkpoint": state["checkpoint"], "results": str(summary.parent),
                "reason": entry.get("reason", "")})
    write_json(directory / "epoch_results.json", {"epochs": states, "rows": rows})
    if not rows:
        return
    addon = any(state.get("inference_layout") == "bonus_addon_batch" for state in states)
    csv_path = directory / "epoch_results.csv"
    with csv_path.with_suffix(".csv.tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix(".csv.tmp").replace(csv_path)
    lines = ["# Per-epoch cached benchmark results", "",
        "Four actual target-state anchors (bonus included), 15 candidates; maximum acceptance including correction/bonus = 16.",
        "The target processes each pending bonus before drafting; those extra forwards are counted separately.",
        "Concurrency 1, greedy BF16, graphs off, maximum 2048 new tokens by default.",
        "Throughput includes prefill and excludes model load/warmup. Task accuracy and parity are not measured.",
        "Evaluation runs on a separate NPU concurrently with training; host contention may affect throughput.", "",
        "| Epoch | Step | Benchmark | Status | Prompts | Acceptance + bonus | Tokens/s | Extra target forwards |",
        "|---:|---:|---|---|---:|---:|---:|---:|"]
    if addon:
        limits = sorted({state.get("concurrency", 32) for state in states})
        lines[2:5] = [
            "Round one: 4 target-state anchors + known bonus embedding + 15 masks (20 queries).",
            "Round two: 4 first-pass drafter-state anchors + 15 masks (19 queries). No extra target forward.",
            f"Concurrency limits: {limits}. Greedy BF16, graphs off; throughput is aggregate tokens/s."]
    def fmt(v):
        return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[k]) for k in ("epoch", "step", "benchmark", "status",
                      "prompts", "acceptance_including_bonus", "tokens_per_second", "extra_target_forwards")) + " |")
    lines += ["", "Checkpoint paths and result directories: `epoch_results.csv` and `epoch_results.json`.", ""]
    tmp = directory / "epoch_results.md.tmp"
    tmp.write_text("\n".join(lines))
    tmp.replace(directory / "epoch_results.md")


def suite_command(args, output, draft):
    layout = getattr(args, "inference_layout", "target_four")
    script = "run_batched_benchmarks.py" if layout == "bonus_addon_batch" else "run_cached_benchmarks.py"
    cmd = [sys.executable, str(Path(args.code_root) / "four_anchor/scripts" / script),
        "--root", args.root, "--code-root", args.code_root, "--inference-layout", layout,
        "--run", args.run, "--model", args.model, "--draft", str(draft),
        "--output", str(output), "--npu", str(args.npu), "--limit", str(args.limit),
        "--max-new-tokens", str(args.max_new_tokens), "--datasets", "all"]
    if layout == "bonus_addon_batch":
        cmd += ["--concurrency", str(getattr(args, "concurrency", 32))]
    if getattr(args, "cache_dir", None):
        cmd += ["--cache-dir", args.cache_dir]
    return cmd


def worker(args):
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prompts = directory / "prepared/plan.json"
        if not prompts.exists():
            raise ValueError("Prepare frozen benchmark prompts before starting the worker")
        while True:
            for path in sorted((directory / "queue").glob("epoch-*.json")):
                job = json.loads(path.read_text())
                state_path = directory / "states" / path.name
                if state_path.exists():
                    previous = json.loads(state_path.read_text())
                    if any(previous[k] != job[k] for k in job):
                        raise ValueError("Checkpoint identity changed for an existing evaluation")
                    if previous["status"] == "completed" or (previous["status"] == "failed" and not args.retry_failed):
                        continue
                attempt = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                output = directory / f"epoch-{job['epoch']:03d}" / attempt
                output.parent.mkdir(parents=True, exist_ok=True)
                state = dict(job, output=str(output), status="running", concurrent_training=True,
                             inference_layout=getattr(args, "inference_layout", "target_four"),
                             concurrency=getattr(args, "concurrency", 1))
                write_json(state_path, state)
                report(directory)
                try:
                    validate_checkpoint(job)
                    command = suite_command(args, output, job["checkpoint"]) + ["--prompt-plan", str(prompts)]
                    with (output.parent / f"{attempt}.log").open("w") as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        try:
                            while child.poll() is None:
                                report(directory)
                                time.sleep(2)
                        finally:
                            if child.poll() is None:
                                child.terminate()
                                child.wait()
                    if child.returncode:
                        raise RuntimeError(f"Benchmark suite exited {child.returncode}; see {log.name}")
                    state["status"] = "completed"
                except Exception as exc:
                    state.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                write_json(state_path, state)
                report(directory)
                print(json.dumps({"event": "epoch_benchmark_finished", **state}), flush=True)
            if (directory / "training_finished.json").exists() or args.once:
                break
            time.sleep(2)
        states = [json.loads(p.read_text()) for p in (directory / "states").glob("epoch-*.json")]
        return int(any(s["status"] != "completed" for s in states))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "worker", "finish"])
    parser.add_argument("--directory", required=True)
    for name in ("root", "code-root", "run", "model"):
        parser.add_argument("--" + name)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--prompt-plan")
    parser.add_argument("--inference-layout", choices=["target_four", "bonus_addon_batch"], default="target_four")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--cache-dir")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    directory = Path(args.directory).resolve()
    args.directory = str(directory)
    if args.command == "finish":
        write_json(directory / "training_finished.json", {"training_exit_code": args.exit_code})
        return 0
    if not all((args.root, args.code_root, args.run, args.model)):
        parser.error("root, code-root, run and model are required")
    if args.command == "worker":
        return worker(args)
    if (directory / "prepared").exists():
        raise ValueError("Use a fresh run/evaluation directory; frozen prompts already exist")
    command = suite_command(args, directory / "prepared", Path(args.run) / "pending-checkpoint") + ["--prepare-only"]
    if args.prompt_plan:
        command += ["--prompt-plan", args.prompt_plan]
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
