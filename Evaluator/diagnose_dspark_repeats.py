#!/usr/bin/env python3
"""Read existing DSpark runs; diagnose timing variation without running inference.

Standard library only. Input suites and model/evaluator sources are never changed.
"""

import argparse
import copy
import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def stats(values):
    if not values:
        return {"mean": None, "stdev": None, "cv_pct": None, "min": None, "max": None}
    avg = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {"mean": avg, "stdev": sd, "cv_pct": 100 * sd / avg if sd is not None and avg else None,
            "min": min(values), "max": max(values)}


def quantile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def normalized_settings(report):
    settings = copy.deepcopy(report.get("comparison_settings"))
    if settings and settings.get("runtime_identity"):
        confidence = (settings["runtime_identity"].get("additional_config") or {}).get("dspark_confidence_config")
        if confidence is not None:
            confidence.pop("mode", None)
            confidence.pop("trace_directory", None)
    return settings


def identity_issues(a, b):
    problems = []
    for key in ("manifest_sha256", "eval_code_sha256"):
        if not a.get(key) or a.get(key) != b.get(key):
            problems.append(f"Missing or different {key}")
    if not a.get("comparison_settings") or normalized_settings(a) != normalized_settings(b):
        problems.append("Target, sampling or runtime settings differ or are missing")
    sa, sb = a.get("server_manifest", {}), b.get("server_manifest", {})
    for key in ("target_sha256", "draft_sha256", "source_sha256", "block_size", "proposals",
                "max_num_seqs", "max_batched_tokens", "memory_utilization", "seed", "kv_block_size",
                "target_execution_requested", "draft_execution_requested", "dspark_head_graph"):
        if sa.get(key) is None or sa.get(key) != sb.get(key):
            problems.append(f"Missing or different server identity: {key}")
    return problems


def generations(path, cell):
    """Validate each saved row and retain only lengths/timings, never generated text."""
    relative = cell.get("generations_file")
    if not relative or not (path.parent / relative).is_file():
        return {"available": False, "note": "Saved generations missing; cannot inspect per-request lengths"}
    lengths, latencies, longest, capped = [], [], [], 0
    with gzip.open(path.parent / relative, "rt", encoding="utf-8") as source:
        for i, line in enumerate(source):
            row = json.loads(line)
            if i >= len(cell["output_hashes"]) or row.get("index") != i or row.get("prompt_sha256") != cell["prompt_hashes"][i]:
                raise ValueError(f"Generation request ordering/hash mismatch: {path}, index {i}")
            tokens = row["output_token_ids"]
            actual = hashlib.sha256(json.dumps([tokens, row["finish_reason"]], separators=(",", ":")).encode()).hexdigest()
            if actual != cell["output_hashes"][i] or row.get("completion_tokens") != len(tokens):
                raise ValueError(f"Generation content/hash mismatch: {path}, index {i}")
            latency = row["latency_seconds"]
            if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"Invalid saved latency: {path}, index {i}")
            lengths.append(len(tokens))
            latencies.append(latency)
            capped += row["finish_reason"] == "length"
            longest.append({"index": i, "output_tokens": len(tokens), "latency_seconds": latency})
    summary = cell["summary"]
    if len(lengths) != summary["requests"] or sum(lengths) != summary["output_tokens"]:
        raise ValueError(f"Saved generations disagree with request/token totals: {path}")
    if not math.isclose(sum(latencies), summary["summed_request_seconds"], rel_tol=1e-7, abs_tol=1e-5):
        raise ValueError(f"Saved generations disagree with total request time: {path}")
    if summary.get("length_limited_requests") is not None and capped != summary["length_limited_requests"]:
        raise ValueError(f"Saved generations disagree with length-limited count: {path}")
    return {"available": True, "output_lengths": lengths,
            "length_p50": quantile(lengths, 0.5), "length_p95": quantile(lengths, 0.95),
            "length_max": max(lengths), "latency_p95_seconds": quantile(latencies, 0.95),
            "latency_max_seconds": max(latencies),
            "longest_request_latencies": sorted(longest, key=lambda r: r["latency_seconds"], reverse=True)[:5]}


def load_group(suite, mode, label):
    index = suite / f"runs_{mode}.json"
    paths = read(index)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"Empty or duplicate completed-run index: {index}")
    records = []
    for repeat, relative in enumerate(paths, 1):
        path = (suite / relative).resolve()
        report = read(path)
        confidence = report.get("server_manifest", {}).get("dspark_confidence", {})
        if not report.get("complete") or confidence.get("mode") != mode or confidence.get("trace_directory"):
            raise ValueError(f"Incomplete, wrong-mode or traced run: {path}")
        cells = {}
        for cell in report["cells"]:
            key = (cell["benchmark"], cell["temperature"], cell["concurrency"])
            if key in cells:
                raise ValueError(f"Duplicate benchmark cell: {path}, {key}")
            summary = cell["summary"]
            n = summary["requests"]
            if n <= 0 or any(len(cell.get(field, [])) != n for field in ("prompt_hashes", "output_hashes")):
                raise ValueError(f"Missing prompt/output hashes: {path}, {key}")
            for field in ("output_tokens", "wall_seconds", "summed_request_seconds", "wall_tokens_per_second", "tokens_per_second"):
                value = summary.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid {field}: {path}, {key}")
            for metric, denominator in (("wall_tokens_per_second", "wall_seconds"), ("tokens_per_second", "summed_request_seconds")):
                if not math.isclose(summary[metric], summary["output_tokens"] / summary[denominator], rel_tol=1e-7):
                    raise ValueError(f"Throughput accounting mismatch: {path}, {key}")
            cells[key] = {"cell": cell, "generations": generations(path, cell)}
        records.append({"group": label, "repeat": repeat, "path": str(path), "report": report, "cells": cells})
    return records


def compare(a, b, key):
    problems = identity_issues(a["report"], b["report"])
    result = {"a": f"{a['group']}:r{a['repeat']}", "b": f"{b['group']}:r{b['repeat']}"}
    if key not in a["cells"] or key not in b["cells"]:
        return {**result, "issues": problems + ["Missing benchmark cell"], "output_agreement_pct": None}
    ac, bc = (record["cells"][key]["cell"] for record in (a, b))
    if key[1] != 0:
        problems.append("Not greedy decoding")
    if ac.get("sampling") != bc.get("sampling"):
        problems.append("Cell sampling parameters differ")
    for name, cell in (("a", ac), ("b", bc)):
        if not cell["summary"].get("measurement_valid"):
            problems.append(f"{name}: measurement not isolated or counters inconsistent")
    aligned = ac["prompt_hashes"] == bc["prompt_hashes"]
    if not aligned:
        problems.append("Ordered prompts differ")
    agreement, differences = None, []
    if aligned:
        differences = [i for i, (x, y) in enumerate(zip(ac["output_hashes"], bc["output_hashes"])) if x != y]
        agreement = 100 * (1 - len(differences) / len(ac["output_hashes"]))
    sa, sb = ac["summary"], bc["summary"]
    return {**result, "issues": problems, "output_agreement_pct": agreement,
            "different_output_indices": differences,
            "output_tokens_change_pct": 100 * (sb["output_tokens"] / sa["output_tokens"] - 1),
            "observed_wall_tps_ratio": sb["wall_tokens_per_second"] / sa["wall_tokens_per_second"] if not problems else None,
            "strict_output_comparison_valid": not problems and agreement == 100}


def build(groups):
    per_run, summaries, comparisons = [], [], []
    keys = sorted({key for records in groups.values() for record in records for key in record["cells"]})
    for key in keys:
        benchmark, temperature, concurrency = key
        for label, records in groups.items():
            rows = []
            for record in records:
                if key not in record["cells"]:
                    continue
                item = record["cells"][key]
                s, g = item["cell"]["summary"], item["generations"]
                al, rate = s.get("acceptance_length_with_bonus_pooled"), s.get("draft_acceptance_rate")
                row = {"benchmark": benchmark, "temperature": temperature, "concurrency": concurrency,
                       "group": label, "repeat": record["repeat"], "requests": s["requests"],
                       "output_tokens": s["output_tokens"], "wall_seconds": s["wall_seconds"],
                       "wall_tokens_per_second": s["wall_tokens_per_second"],
                       "average_client_in_flight": s["summed_request_seconds"] / s["wall_seconds"],
                       "tpot_ms": s.get("tpot_ms_request_mean"), "acceptance_length": al,
                       "mean_verified_proposals": (al - 1) / rate if al is not None and rate else None,
                       "length_limited_requests": s.get("length_limited_requests"),
                       "length_p95": g.get("length_p95"), "length_max": g.get("length_max"),
                       "latency_max_seconds": g.get("latency_max_seconds"),
                       "measurement_valid": s.get("measurement_valid"), "generations_checked": g["available"],
                       "run_info": record["path"]}
                rows.append(row)
                per_run.append(row)
            pairs = [compare(a, b, key) for a, b in itertools.combinations(records, 2)]
            agreement = [p["output_agreement_pct"] for p in pairs if p["output_agreement_pct"] is not None]
            summaries.append({"benchmark": benchmark, "temperature": temperature, "concurrency": concurrency,
                              "group": label, "runs": len(rows), "expected_indexed_runs": len(records),
                              "throughput": stats([r["wall_tokens_per_second"] for r in rows]),
                              "output_tokens": stats([r["output_tokens"] for r in rows]),
                              "wall_seconds": stats([r["wall_seconds"] for r in rows]),
                              "same_mode_output_agreement_pct": stats(agreement), "same_mode_pairs": pairs})
        for left, right in itertools.combinations(groups, 2):
            pairs = [compare(a, b, key) for a in groups[left] for b in groups[right]]
            a_values = [r["cells"][key]["cell"]["summary"]["wall_tokens_per_second"] for r in groups[left] if key in r["cells"]]
            b_values = [r["cells"][key]["cell"]["summary"]["wall_tokens_per_second"] for r in groups[right] if key in r["cells"]]
            valid_settings = bool(pairs) and all(not p["issues"] for p in pairs)
            agreement = [p["output_agreement_pct"] for p in pairs if p["output_agreement_pct"] is not None]
            ratio = statistics.mean(b_values) / statistics.mean(a_values) if valid_settings else None
            comparisons.append({"benchmark": benchmark, "temperature": temperature, "concurrency": concurrency,
                                "baseline": left, "candidate": right, "settings_compatible": valid_settings,
                                "observed_ratio_of_means": ratio,
                                "strict_ratio_of_means": ratio if all(p.get("strict_output_comparison_valid") for p in pairs) else None,
                                "output_agreement_pct": stats(agreement), "all_repeat_pairs": pairs})
    return {"schema": "dspark_repeat_diagnostics_v1", "per_run": per_run,
            "groups": summaries, "comparisons": comparisons,
            "notes": ["Descriptive diagnostics only: no significance test or causal claim about the confidence head or training.",
                      "Cross-group comparisons include every repeat combination; repeats are not time-matched pairs.",
                      "Average client in-flight requests equals summed request seconds / wall seconds, not NPU batch size.",
                      "No request start/end timestamps were saved; a batch occupancy timeline or exact drain time cannot be reconstructed.",
                      "Missing generation files limit length diagnostics; output hashes can still be compared.",
                      "Read-only analysis of saved results. No server, model, torch, NPU, or network access."]}


def fmt(value, digits=2):
    return "NA" if value is None else f"{value:.{digits}f}"


def text_report(report):
    lines = ["DSpark saved-run diagnostics", "", "PER REPEAT (wall TPS, client overlap and acceptance; no new inference)",
             "dataset group repeat tokens wall_s wall_TPS avg_inflight AL avg_K capped p95_len max_len"]
    for r in report["per_run"]:
        lines.append(f"{r['benchmark']} {r['group']} r{r['repeat']} {r['output_tokens']} {fmt(r['wall_seconds'])} "
                     f"{fmt(r['wall_tokens_per_second'])} {fmt(r['average_client_in_flight'])} {fmt(r['acceptance_length'], 4)} "
                     f"{fmt(r['mean_verified_proposals'], 3)} {r['length_limited_requests']} {fmt(r['length_p95'], 0)} {r['length_max']}")
    lines += ["", "WITHIN EACH GROUP (CV = standard deviation / mean; agreement spans all repeat pairs)",
              "dataset group runs TPS_mean TPS_sd CV_pct output_agreement_min_pct output_agreement_max_pct"]
    for r in report["groups"]:
        t, a = r["throughput"], r["same_mode_output_agreement_pct"]
        lines.append(f"{r['benchmark']} {r['group']} {r['runs']} {fmt(t['mean'])} {fmt(t['stdev'])} {fmt(t['cv_pct'])} {fmt(a['min'])} {fmt(a['max'])}")
    lines += ["", "BETWEEN GROUPS (ratio = candidate mean TPS / baseline mean TPS)",
              "dataset baseline candidate observed_ratio strict_ratio agreement_min_pct agreement_max_pct settings_match"]
    for r in report["comparisons"]:
        a = r["output_agreement_pct"]
        lines.append(f"{r['benchmark']} {r['baseline']} {r['candidate']} {fmt(r['observed_ratio_of_means'], 4)} "
                     f"{fmt(r['strict_ratio_of_means'], 4)} {fmt(a['min'])} {fmt(a['max'])} {r['settings_compatible']}")
    issues = sorted({issue for r in report["groups"] for p in r["same_mode_pairs"] for issue in p["issues"]}
                   | {issue for r in report["comparisons"] for p in r["all_repeat_pairs"] for issue in p["issues"]})
    if issues:
        lines += ["", "COMPATIBILITY / MEASUREMENT ISSUES:"] + issues
    missing = sum(not row["generations_checked"] for row in report["per_run"])
    lines += ["", f"Cells missing saved generation files: {missing}"]
    lines += ["", "INTERPRETATION:",
              "Output disagreement within OFF or within ON means the workload itself varied across repeats.",
              "If within-mode outputs agree but cross-mode outputs do not, investigate cutting/output correctness.",
              "TPS variation comparable to the proposed improvement makes that improvement inconclusive.",
              "Different client overlap or length distributions can affect wall throughput independently of per-token latency.",
              "Same-mode drift does not prove cross-mode drift harmless. Check first divergence if outputs differ.",
              "NA strict ratio means no identical-output-validated comparison; inspect diagnostic.json for details."]
    lines += ["", *report["notes"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-suite", type=Path, required=True, help="Earlier suite containing OFF and CUT")
    parser.add_argument("--rerun-suite", type=Path, required=True, help="New suite containing CUT repeats")
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent diagnostic directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    groups = {}
    for label, path, mode in (("off", args.original_suite, "off"), ("old_on", args.original_suite, "cut"),
                              ("new_on", args.rerun_suite, "cut")):
        print(f"Checking {label}: {path}", flush=True)
        groups[label] = load_group(path, mode, label)
    report = build(groups)
    rendered = text_report(report)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "diagnostic.txt").write_text(rendered)
    with (args.output / "per_run.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["per_run"][0]))
        writer.writeheader()
        writer.writerows(report["per_run"])
    print(rendered)
    print(f"Saved diagnostic: {args.output.resolve() / 'diagnostic.txt'}")


if __name__ == "__main__":
    main()
