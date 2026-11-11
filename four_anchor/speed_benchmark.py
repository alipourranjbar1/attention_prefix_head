"""Compare isolated real training runs, preserving the data and objective.

No remote access, process killing, or active-checkpoint changes. Each trial
starts from the same seed and runs a bounded number of full-sized updates.
Teacher startup, graph capture, and model loading are excluded from steady
step timings, but are included in recorded trial wall time.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time


DEFAULTS = {"TARGET_NPUS": "0,1", "TRAIN_NPUS": "2,3,4,5,6,7", "SEQ_LENGTH": "9216",
    "MAX_ANCHORS": "1536", "EPOCHS": "5", "LR": "0.0006", "GRADIENT_ACCUMULATION": "1",
    "TRAIN_DATA_RATIO": "1.0", "BOUNDARY_CONFIDENCE": "confidence_head", "BOUNDARY_EXPLORATION": "0.5",
    "CONFIDENCE_ALPHA": "1.0", "LOSS_FN": '{"ce":0.1,"tv":0.9}', "LOSS_GRAPH": "off",
    "EVAL_EACH_EPOCH": "0", "NUM_WORKERS": "4", "SEED": "42", "BOUNDARY_BETA": "0.7"}
BASELINE = {"HEAD_LOSS_BACKEND": "checkpoint", "GRADIENT_CHECKPOINTING": "1",
            "BLOCKS_PER_CHUNK": "128", "HEAD_CHUNK_SIZE": "256", "DISTRIBUTED_STRATEGY": "fsdp",
            "TARGET_EXECUTION": "eager"}
BASELINE["PROPOSAL_BASE_BATCHED"] = "0"


def graph_observed(text):
    # Count real scheduler dispatches, not configuration strings or warmup.
    rows = re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(?:CUDAGraphMode\.)?PIECEWISE\s*\|\s*(\d+)\s*\|", text)
    return sum(int(count) for real, _, _, count in rows if int(real) > 1)


def summarize(path, warmup, expected_steps):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or rows[-1]["step"] != expected_steps:
        raise ValueError("Trial did not complete all requested steps")
    rows = [r for r in rows if r["step"] > warmup]
    values = [r["max_rank_step_seconds"] for r in rows]
    if len(values) < 5 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Not enough valid steady-step timings")
    return {"median_step_seconds": statistics.median(values), "mean_step_seconds": statistics.mean(values),
            "min_step_seconds": min(values), "max_step_seconds": max(values), "measured_steps": len(values),
            "last_loss": rows[-1]["loss"], "peak_rank0_allocated_gib": max(r["rank0_peak_memory_allocated_gib"] for r in rows),
            "phase_mean_seconds": {k: statistics.mean(r["max_rank_" + k + "_seconds"] for r in rows)
                                   for k in ("input", "forward", "backward", "optimizer")}}


def report(output, records, selected):
    (output / "results.json").write_text(json.dumps({"trials": records, "selected": selected}, indent=2) + "\n")
    lines = ["# DSpark training speed trials", "", "Same 9,216-token/rank budget, 1,536 sampled blocks/rank, two passes and loss.",
        "Timings are maximum-rank update latency, with synchronization enabled consistently in every trial.",
        "These are short hardware trials, not guarantees of full-epoch speed or identical floating-point trajectories.", "",
        "| Trial | Status | Median s/update | Peak allocated GiB (rank 0) | Teacher graph dispatches |",
        "|---|---|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['name']} | {r['status']} | {r.get('median_step_seconds', '—')} | {r.get('peak_rank0_allocated_gib', '—')} | {r.get('prefill_graph_dispatches', 0)} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run(root, output, steps=24, warmup=8):
    if not 1 <= warmup < steps <= 100 or steps - warmup < 5:
        raise ValueError("Use 1 <= warmup < steps <= 100 with at least five measured steps")
    if os.environ.get("RESUME"):
        raise ValueError("Unset RESUME: trials are separate fresh runs")
    if output.exists():
        raise ValueError(f"Choose a new benchmark directory: {output}")
    output.mkdir(parents=True)
    env = {**os.environ, **DEFAULTS, "DATA_CACHE_DIR": str(root / "output/.four_anchor_data_cache"),
           "MAX_STEPS": str(steps), "PROFILE_STEPS": str(steps), "BENCHMARK_ONLY": "1",
           "CHECK_ONLY": "0", "HARDWARE_CHECK_ONLY": "0", "LOG_FREQ": "1"}
    # Do not let stale command-line overrides alter this comparison.
    env.pop("DFLASH_INIT", None)
    records, best = [], None

    def trial(name, settings):
        nonlocal best
        path = output / name
        start = time.monotonic()
        print(f"Trial {name}: {settings}", flush=True)
        print(f"Live trial log: {output / (name + '.log')}", flush=True)
        with (output / (name + ".log")).open("w") as log:
            result = subprocess.run(["bash", str(root / "four_anchor/scripts/train_dspark_ascend.sh")],
                                    cwd=root, env={**env, **settings, "OUTPUT_DIR": str(path)},
                                    stdout=log, stderr=subprocess.STDOUT)
        rec = {"name": name, "settings": settings, "exit_code": result.returncode,
               "wall_seconds_including_startup": time.monotonic() - start,
               "log": str(output / (name + ".log")), "status": "failed"}
        if result.returncode == 0:
            try:
                rec.update(summarize(path / "checkpoints/metrics.jsonl", warmup, steps))
                rec["prefill_graph_dispatches"] = graph_observed((path / "target_server.log").read_text(errors="replace"))
                if settings["TARGET_EXECUTION"] == "graph" and rec["prefill_graph_dispatches"] == 0:
                    raise ValueError("Requested target graphs but no PIECEWISE prefill dispatch was observed")
                rec["status"] = "completed"
                if best is None or rec["median_step_seconds"] < best["median_step_seconds"] * .97:
                    best = rec
            except (OSError, ValueError, KeyError) as exc:
                rec["reason"] = str(exc)
        records.append(rec)
        report(output, records, best)
        print(json.dumps(rec), flush=True)
        return rec

    baseline = trial("01_reference", dict(BASELINE))
    if baseline["status"] != "completed":
        raise RuntimeError(f"Reference run failed; no further trials started. See {baseline['log']}")
    # Adaptive comparisons retain faster tested settings. FSDP remains in use
    # unless a DDP trial fits and actually improves measured update time.
    for name, change in [
        ("02_cached_head", {"HEAD_LOSS_BACKEND": "cached_vjp"}),
        ("02b_batched_proposal_base", {"PROPOSAL_BASE_BATCHED": "1"}),
        ("03_without_backbone_recompute", {"GRADIENT_CHECKPOINTING": "0"}),
        ("04_head_chunks_512", {"HEAD_CHUNK_SIZE": "512"}),
        ("05_block_chunks_64", {"BLOCKS_PER_CHUNK": "64"}),
        ("06_ddp", {"DISTRIBUTED_STRATEGY": "ddp"}),
        ("07_target_graph", {"TARGET_EXECUTION": "graph"}),
    ]:
        trial(name, {**best["settings"], **change})
    # Repeat the selected configuration; include confirmation results in the
    # report. A single noisy result is not silently advertised as a speedup.
    selected = dict(best)
    confirmation = trial("08_confirm_selected", dict(selected["settings"]))
    if confirmation["status"] != "completed":
        raise RuntimeError("Selected settings failed confirmation; no training command selected")
    chosen = {"settings": selected["settings"], "reference_seconds": baseline["median_step_seconds"],
              "confirmation_seconds": confirmation["median_step_seconds"],
              "measured_speedup": baseline["median_step_seconds"] / confirmation["median_step_seconds"]}
    if selected["name"] != baseline["name"] and confirmation["median_step_seconds"] >= baseline["median_step_seconds"] * .97:
        chosen.update(settings=baseline["settings"], measured_speedup=1.,
                      note="Candidate improvement did not repeat by at least 3%; retained reference settings")
    (output / "selected.json").write_text(json.dumps(chosen, indent=2) + "\n")
    lines = ["# Sourced settings from confirmed short trials. No process is started by this file."]
    for k, v in {**DEFAULTS, **chosen["settings"], "BENCHMARK_ONLY": "0", "PROFILE_STEPS": "10"}.items():
        lines.append("export " + k + "=" + shlex.quote(v))
    lines.append("unset MAX_STEPS RESUME OUTPUT_DIR")
    (output / "selected.env").write_text("\n".join(lines) + "\n")
    report(output, records, chosen)
    print(f"Finished. Results: {output / 'summary.md'}\nSettings: {output / 'selected.env'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--warmup", type=int, default=8)
    args = p.parse_args()
    run(Path(__file__).resolve().parents[1], args.output.resolve(), args.steps, args.warmup)
