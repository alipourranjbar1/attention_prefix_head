#!/usr/bin/env python3
"""Summarize confidence accuracy or compare matched DSpark HTTP benchmarks."""

import argparse
import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def length_accuracy(pairs):
    errors = [predicted - actual for predicted, actual in pairs]
    n = len(errors)
    return {
        "rounds": n,
        "exact_pct": 100 * sum(x == 0 for x in errors) / n if n else None,
        "under_pct": 100 * sum(x < 0 for x in errors) / n if n else None,
        "over_pct": 100 * sum(x > 0 for x in errors) / n if n else None,
        "within_one_pct": 100 * sum(abs(x) <= 1 for x in errors) / n if n else None,
        "mae": mean([abs(x) for x in errors]),
        "signed_error": mean(errors),
    }


def accuracy(rows):
    verified = [r for r in rows if r["status"] == "verified"]
    full = [r for r in verified if r["full_block_verified"]]
    if any(r["actual_accepted_full"] is None for r in full):
        raise ValueError("A full-block label is unexpectedly censored")
    paired = [(r["predicted_accepted"], r["actual_accepted_full"]) for r in full]
    budget = [(r["selected_verify_tokens"], r["actual_accepted_full"]) for r in full]
    brier = []
    for row in full:
        survival = 1.0
        for position, probability in enumerate(row["conditional_probabilities"]):
            survival *= probability
            brier.append((survival - (position < row["actual_accepted_full"])) ** 2)
    return {
        "modes": sorted({r["mode"] for r in rows}),
        "matched_trace_rounds": len(rows),
        "skipped_reasons": dict(
            Counter(r["reason"] for r in rows if r["status"] == "skipped")
        ),
        "full_block_accuracy": length_accuracy(paired),
        "selected_budget_vs_actual_full_length": length_accuracy(budget),
        "before_cut": {
            "mean_predicted_acceptance_length": mean(
                [r["predicted_acceptance_length"] for r in full]
            ),
            "mean_expected_acceptance_length": mean(
                [r["expected_acceptance_length"] for r in full]
            ),
            "mean_actual_acceptance_length": mean(
                [r["actual_acceptance_length_full"] for r in full]
            ),
            "survival_brier_score": mean(brier),
        },
        "counterfactual_cut_on_same_full_blocks": {
            "rounds": len(full),
            "mean_selected_verify_tokens": mean(
                [r["selected_verify_tokens"] for r in full]
            ),
            "mean_acceptance_length": mean(
                [
                    1 + min(r["actual_accepted_full"], r["selected_verify_tokens"])
                    for r in full
                ]
            ),
            "mean_accepted_tokens_lost": mean(
                [
                    max(0, r["actual_accepted_full"] - r["selected_verify_tokens"])
                    for r in full
                ]
            ),
            "capped_prediction_accuracy": length_accuracy(
                [
                    (
                        min(r["predicted_accepted"], r["selected_verify_tokens"]),
                        min(r["actual_accepted_full"], r["selected_verify_tokens"]),
                    )
                    for r in full
                ]
            ),
            "note": "Same observed prefixes only; this is not the acceptance or speed of a separate cut-mode generation.",
        },
        "observed_after_verification": {
            "rounds": len(verified),
            "mean_acceptance_length": mean(
                [r["acceptance_length_after_cut"] for r in verified]
            ),
            "mean_verified_draft_tokens": mean(
                [r["verified_tokens"] for r in verified]
            ),
            "full_length_censored_rounds": sum(r["censored"] for r in verified),
            "known_shortened_boundaries": sum(
                r["label_complete"] and not r["full_block_verified"] for r in verified
            ),
            "scheduler_clipped_rounds": sum(r["scheduler_clipped"] for r in verified),
            "capped_prediction_accuracy": length_accuracy(
                [
                    (
                        min(r["predicted_accepted"], r["verified_tokens"]),
                        r["accepted_after_cut"],
                    )
                    for r in verified
                ]
            ),
        },
        "note": "Lengths include one target correction/bonus. Full-block accuracy excludes EOS and shortened blocks. "
        "In cut mode this is a policy-selected subset; use observe mode for overall head accuracy. "
        "Capping predictions and labels makes accuracy easier; use full_block_accuracy to assess the head.",
    }


def accuracy_report(trace_directory, report):
    if not report.get("complete"):
        raise ValueError("Evaluation did not complete")
    cells = report["cells"]
    owners = {}
    for index, cell in enumerate(cells):
        request_ids = cell.get("request_ids")
        if not request_ids or any(not rid for rid in request_ids):
            raise ValueError("Run the patched evaluator to retain measured request IDs")
        # /v1/completions with one prompt and n=1 uses engine ID <API ID>-0.
        for rid in request_ids:
            key = rid + "-0"
            if key in owners:
                raise ValueError("Duplicate measured request ID")
            owners[key] = index
    grouped = [[] for _ in cells]
    files = sorted(Path(trace_directory).glob("rounds_*.jsonl"))
    if not files:
        raise ValueError("No confidence traces found")
    unmatched = 0
    seen = set()
    for path in files:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["request_id"] not in owners:
                    unmatched += 1
                    continue
                key = (row["request_id"], row["round_id"])
                if key in seen:
                    raise ValueError("Duplicate confidence trace record")
                seen.add(key)
                grouped[owners[row["request_id"]]].append(row)
    if not any(grouped):
        raise ValueError(
            "No traces match measured request IDs; do not use warmup as benchmark accuracy"
        )
    return {
        "schema": "dspark_confidence_accuracy_v1",
        "excluded_unmeasured_rounds": unmatched,
        "cells": [
            dict(
                benchmark=c["benchmark"],
                concurrency=c.get("concurrency", 1),
                temperature=c["temperature"],
                **accuracy(rows),
            )
            for c, rows in zip(cells, grouped)
        ],
    }


def normalized_settings(report):
    settings = copy.deepcopy(report.get("comparison_settings"))
    if settings and settings.get("runtime_identity"):
        runtime = settings["runtime_identity"]
        confidence = runtime.get("additional_config", {}).get(
            "dspark_confidence_config"
        )
        if confidence is not None:
            confidence.pop("mode", None)
            confidence.pop("trace_directory", None)
    return settings


def compare_reports(baseline, candidate, expected_mode):
    issues = []
    base_server, server = (
        baseline.get("server_manifest") or {},
        candidate.get("server_manifest") or {},
    )
    for report, manifest, mode in (
        (baseline, base_server, "off"),
        (candidate, server, expected_mode),
    ):
        if not report.get("complete"):
            issues.append(f"{mode} evaluation incomplete")
        confidence = manifest.get("dspark_confidence") or {}
        if confidence.get("mode") != mode:
            issues.append(f"Expected confidence mode {mode}")
        if confidence.get("trace_directory"):
            issues.append(
                f"{mode} has diagnostic tracing enabled; timing is not comparable"
            )
    if normalized_settings(baseline) != normalized_settings(candidate):
        issues.append(
            "Target, decoding or runtime settings differ beyond the confidence mode"
        )
    for key in (
        "target_sha256",
        "draft_sha256",
        "source_sha256",
        "block_size",
        "proposals",
        "max_num_seqs",
        "max_batched_tokens",
        "memory_utilization",
        "seed",
        "kv_block_size",
        "target_execution_requested",
        "draft_execution_requested",
        "dspark_head_graph",
    ):
        if base_server.get(key) is None or base_server.get(key) != server.get(key):
            issues.append(f"Missing or differing server identity: {key}")
    if not baseline.get("manifest_sha256") or baseline.get(
        "manifest_sha256"
    ) != candidate.get("manifest_sha256"):
        issues.append("Frozen prompt manifests differ")
    if not baseline.get("eval_code_sha256") or baseline.get(
        "eval_code_sha256"
    ) != candidate.get("eval_code_sha256"):
        issues.append("Evaluator source differs")
    baseline_cells = {
        (c["benchmark"], c["temperature"], c.get("concurrency", 1)): c
        for c in baseline["cells"]
    }
    output = []
    for cell in candidate["cells"]:
        key = (cell["benchmark"], cell["temperature"], cell.get("concurrency", 1))
        previous = baseline_cells.get(key)
        errors = list(issues)
        summary = cell["summary"]
        if previous is None:
            errors.append("Baseline cell missing")
        else:
            old = previous["summary"]
            if key[1] != 0:
                errors.append("Confidence experiment requires greedy decoding")
            for field in ("prompt_hashes", "output_hashes"):
                if not previous.get(field) or previous.get(field) != cell.get(field):
                    errors.append(f"{field} differ or are missing")
            if cell.get("sampling") != previous.get("sampling"):
                errors.append("Sampling parameters differ")
            if not summary.get("measurement_valid") or not old.get("measurement_valid"):
                errors.append("Measurement not isolated or counters inconsistent")
            if summary.get("requests") != old.get("requests") or summary.get(
                "requests"
            ) != len(cell.get("output_hashes", [])):
                errors.append("Completed request counts differ")
            for value in (
                summary.get("tokens_per_second"),
                old.get("tokens_per_second"),
            ):
                if value is None or not math.isfinite(value) or value <= 0:
                    errors.append("Invalid throughput")
        valid = not errors
        output.append(
            {
                "mode": expected_mode,
                "benchmark": key[0],
                "temperature": key[1],
                "concurrency": key[2],
                "comparison_valid": valid,
                "issues": errors,
                "baseline_tokens_per_second": previous["summary"].get(
                    "tokens_per_second"
                )
                if previous
                else None,
                "tokens_per_second": summary.get("tokens_per_second"),
                "speedup": summary["tokens_per_second"]
                / previous["summary"]["tokens_per_second"]
                if valid
                else None,
                "speedup_percent": 100
                * (
                    summary["tokens_per_second"]
                    / previous["summary"]["tokens_per_second"]
                    - 1
                )
                if valid
                else None,
                "baseline_acceptance_length": previous["summary"].get(
                    "acceptance_length_with_bonus_pooled"
                )
                if previous
                else None,
                "acceptance_length": summary.get("acceptance_length_with_bonus_pooled"),
                "draft_acceptance_rate": summary.get("draft_acceptance_rate"),
                "ttft_ms": summary.get("ttft_ms_mean"),
                "tpot_ms": summary.get("tpot_ms_request_mean"),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("accuracy")
    audit.add_argument("--trace-dir", type=Path, required=True)
    audit.add_argument("--run-info", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--observe", type=Path)
    compare.add_argument("--cut", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "accuracy":
        result = accuracy_report(args.trace_dir, json.loads(args.run_info.read_text()))
    else:
        baseline = json.loads(args.baseline.read_text())
        rows = []
        if args.observe:
            rows += compare_reports(
                baseline, json.loads(args.observe.read_text()), "observe"
            )
        rows += compare_reports(baseline, json.loads(args.cut.read_text()), "cut")
        result = {
            "schema": "dspark_confidence_speed_v1",
            "cells": rows,
            "note": "Speedup is measured throughput / confidence-off throughput, with identical greedy output hashes.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.command == "compare" and result["cells"]:
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["cells"][0]))
            writer.writeheader()
            writer.writerows(result["cells"])
    print(json.dumps(result, indent=2))
    if args.command == "compare" and (
        not result["cells"]
        or any(not row["comparison_valid"] for row in result["cells"])
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
