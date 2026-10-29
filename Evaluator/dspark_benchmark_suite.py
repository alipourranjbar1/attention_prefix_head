#!/usr/bin/env python3
"""Full-split DSpark speed experiments with repeatability and output diagnostics."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import dspark_confidence_report as confidence_report
import evaluator

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = tuple(name for name in evaluator.COUNTS if name != "alpaca")


def read(path):
    return json.loads(Path(path).read_text())


def reusable_datasets(path, target, source_overrides):
    """Reuse complete tokenized splits, never a sampled prompt set."""
    saved = read(path)
    tokenizer_hashes = {
        item.name: evaluator.file_hash(item)
        for item in sorted(target.iterdir())
        if item.name in (
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "special_tokens_map.json",
        )
    }
    if (saved.get("format") != 3 or saved.get("thinking") is not False
            or saved.get("target_config_sha256") != evaluator.file_hash(target / "config.json")
            or not tokenizer_hashes or saved.get("tokenizer_sha256") != tokenizer_hashes):
        raise ValueError(
            f"Cannot reuse {path}: format, thinking mode, target config or tokenizer fingerprint differs"
        )
    overridden = {value.partition("=")[0] for value in source_overrides}
    reusable = {}
    for name in BENCHMARKS:
        if name in overridden or name not in saved["datasets"]:
            continue
        dataset = saved["datasets"][name]
        count = dataset.get("source_rows")
        if not isinstance(count, int) or count <= 0 or len(dataset["results"]) != count:
            print(f"Rebuild {name}: saved manifest is sampled or lacks full source counts", flush=True)
            continue
        evaluator.require_full_manifest(saved, [name])
        evaluator.select_prompts(saved, [name], 0, False)
        if name == "lcb" and saved.get("choices", {}).get("lcb_release") != "release_v5":
            raise ValueError(f"Cannot reuse {path}: LiveCodeBench release must be v5")
        reusable[name] = dataset
        print(f"Reused {name}: {count}/{count} saved prompts", flush=True)
    return reusable, {
        "path": str(path.resolve()), "sha256": evaluator.file_hash(path),
        "datasets": list(reusable), "original_choices": saved.get("choices"),
    }


def prepare(args):
    suite = args.suite.resolve()
    concurrency = getattr(args, "concurrency", 64)
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    suite.mkdir(parents=True, exist_ok=True)
    settings = {
        "target": str(args.target.resolve()), "draft": str(args.draft.resolve()),
        "concurrency": concurrency, "proposals": 7, "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens, "seed": 42,
        "benchmarks": list(BENCHMARKS), "thinking": False,
        "evaluator_sha256": evaluator.file_hash(ROOT / "Evaluator/evaluator.py"),
    }
    if not (args.draft / "config.json").is_file():
        raise ValueError(f"Draft checkpoint not found: {args.draft}")
    config = read(args.target / "config.json")
    manifest_path = suite / "prompts.json"
    if (suite / "suite.json").is_file():
        saved = read(suite / "suite.json")
        if saved["settings"] != settings:
            raise ValueError("Existing suite settings/code differ. Set SUITE_DIR to a new directory.")
        validate_suite(suite)
        print_inventory(saved)
        return
    if not manifest_path.exists():
        # Commit a manifest only after every dataset has prepared successfully.
        with tempfile.TemporaryDirectory(prefix="prepare_", dir=suite) as temporary:
            reused, provenance = {}, None
            if getattr(args, "reuse_manifest", None):
                reused, provenance = reusable_datasets(args.reuse_manifest, args.target, args.source)
            options = argparse.Namespace(
                manifest=Path(temporary) / "prompts.json", rebuild_manifest=False,
                thinking=False, target=args.target, source=args.source,
                prompt_seed=42, num_prompts=0, all_prompts=True,
                offline=True, lcb_release=5,
            )
            missing = [name for name in BENCHMARKS if name not in reused]
            manifest = evaluator.prepare_manifest(options, missing)
            manifest["datasets"].update(reused)
            manifest["datasets"] = {name: manifest["datasets"][name] for name in BENCHMARKS}
            if provenance:
                manifest["prepared_manifest_reuse"] = provenance
            evaluator.require_full_manifest(manifest, BENCHMARKS)
            evaluator.write_json(options.manifest, manifest)
            options.manifest.replace(manifest_path)
    manifest = read(manifest_path)
    evaluator.require_full_manifest(manifest, BENCHMARKS)
    evaluator.select_prompts(manifest, BENCHMARKS, 0, False)
    if manifest["target_config_sha256"] != evaluator.file_hash(args.target / "config.json"):
        raise ValueError("Full prompt manifest uses a different target config")
    inventory = {}
    maximum = 0
    for name in BENCHMARKS:
        dataset = manifest["datasets"][name]
        longest = max(len(row["prompt_token_ids"]) for row in dataset["results"])
        maximum = max(maximum, longest)
        inventory[name] = {
            "requests": len(dataset["results"]), "source_rows": dataset["source_rows"],
            "max_prompt_tokens": longest, "source": dataset["source"],
        }
    needed = maximum + args.max_new_tokens + 16
    model_limit = config.get("max_position_embeddings")
    if model_limit and needed > model_limit:
        raise ValueError(
            f"Full dataset needs {needed} context tokens, target config allows {model_limit}. "
            "No prompts were dropped. Review the longest prompts/output budget before proceeding."
        )
    context = max(5120, math.ceil(needed / 128) * 128)
    if model_limit:
        context = min(context, model_limit)
    metadata = {
        "schema": "dspark_full_suite_v1", "settings": settings,
        "manifest_sha256": evaluator.file_hash(manifest_path), "inventory": inventory,
        "max_model_len": context, "max_batched_tokens": context,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Full saved source splits; MT-Bench first user turn; LCB release v5. No task-quality scoring.",
    }
    evaluator.write_json(suite / "suite.json", metadata)
    print_inventory(metadata)


def print_inventory(metadata):
    print("Full source splits (no sampling):", flush=True)
    for name, item in metadata["inventory"].items():
        effective = min(item["requests"], metadata["settings"]["concurrency"])
        print(f"  {name}: {item['requests']}/{item['source_rows']} prompts; at most {effective} in flight", flush=True)
    total = sum(item["requests"] for item in metadata["inventory"].values())
    repeats = metadata["settings"]["repeats"]
    print(f"{total} prompts per pass; {repeats} passes per mode; {2 * repeats * total} measured requests total.", flush=True)
    print(f"Context and unchunked-prefill token budget: {metadata['max_model_len']}.", flush=True)


def validate_suite(suite):
    metadata = read(suite / "suite.json")
    if evaluator.file_hash(suite / "prompts.json") != metadata["manifest_sha256"]:
        raise ValueError("Frozen full-dataset manifest changed")
    if evaluator.file_hash(ROOT / "Evaluator/evaluator.py") != metadata["settings"]["evaluator_sha256"]:
        raise ValueError("Evaluator changed after preparation; use a new SUITE_DIR")
    return metadata


def load_index(suite, mode):
    path = suite / f"runs_{mode}.json"
    return read(path) if path.exists() else []


def evaluate(args):
    suite = args.suite.resolve()
    metadata = validate_suite(suite)
    settings = metadata["settings"]
    requested = getattr(args, "concurrency", None)
    if requested is not None and requested != settings["concurrency"]:
        raise ValueError("Requested concurrency differs from the saved suite; check CONCURRENCY and SUITE_DIR")
    launch_path = args.server_manifest.resolve()
    launch = read(launch_path)
    confidence = launch.get("dspark_confidence") or {}
    mode = confidence.get("mode")
    if mode not in {"off", "cut"} or confidence.get("trace_directory"):
        raise ValueError("Expected a head-off or cut server with diagnostic tracing disabled")
    expected = {
        "kind": "dspark", "proposals": 7, "max_num_seqs": settings["concurrency"],
        "max_model_len": metadata["max_model_len"],
        "max_batched_tokens": metadata["max_batched_tokens"],
    }
    if any(launch.get(key) != value for key, value in expected.items()):
        raise ValueError("Server does not match the full-suite configuration; start serve_dspark_full_suite.sh")
    for key in ("target", "draft"):
        if Path(launch[key]).resolve() != Path(settings[key]).resolve():
            raise ValueError(f"Suite/server {key} mismatch")
    completed = load_index(suite, mode)
    if len(completed) > settings["repeats"]:
        raise ValueError("More saved runs than configured repeats")
    for relative in completed:
        if not read(suite / relative).get("complete"):
            raise ValueError("Saved completed-run index contains an incomplete run")
    for repeat in range(len(completed), settings["repeats"]):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = suite / "runs" / f"{mode}_r{repeat + 1}_{stamp}"
        command = [
            sys.executable, str(ROOT / "Evaluator/evaluator.py"),
            "--base-url", args.base_url, "--server-manifest", str(launch_path),
            "--target", settings["target"], "--method", f"dspark_full_{mode}_r{repeat + 1}",
            "--datasets", *BENCHMARKS, "--manifest", str(suite / "prompts.json"),
            "--all-prompts", "--save-generations", "--offline", "--no-thinking",
            "--temperatures", "0", "--concurrencies", str(settings["concurrency"]), "--seed", "42",
            "--max-new-tokens", str(settings["max_new_tokens"]), "--output", str(output),
        ]
        print(f"\nStarting {mode} repeat {repeat + 1}/{settings['repeats']}: ALL source rows", flush=True)
        subprocess.run(command, check=True)
        run = read(output / "run_info.json")
        if not run.get("complete"):
            raise ValueError("Evaluation incomplete; completed-run index not updated")
        cells = {cell["benchmark"]: cell for cell in run["cells"]}
        for name, item in metadata["inventory"].items():
            if cells[name]["summary"]["requests"] != item["requests"]:
                raise ValueError(f"{name}: evaluation did not cover every source row")
            if cells[name]["concurrency"] != settings["concurrency"]:
                raise ValueError(f"{name}: evaluation used a different concurrency")
        completed.append(str((output / "run_info.json").relative_to(suite)))
        evaluator.write_json(suite / f"runs_{mode}.json", completed)
    result = build_report(suite)
    print(f"\n{mode}: {len(completed)} completed repeats. Report: {suite / 'suite_report.csv'}", flush=True)
    if not result["complete"]:
        print("Start the other server mode, then rerun the same evaluator script.", flush=True)


def stats(values):
    return {
        "runs": len(values), "mean": statistics.mean(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
        "values": values,
    }


def parity(first, second):
    a, b = first.get("output_hashes"), second.get("output_hashes")
    if (not a or not b or len(a) != len(b)
            or first.get("prompt_hashes") != second.get("prompt_hashes")):
        return {"comparable": False, "matching": None, "total": None, "agreement_pct": None, "different_indices": []}
    different = [index for index, (x, y) in enumerate(zip(a, b)) if x != y]
    return {
        "comparable": True, "matching": len(a) - len(different), "total": len(a),
        "agreement_pct": 100 * (len(a) - len(different)) / len(a),
        "different_indices": different,
    }


def first_difference(off, cut):
    a, b = off["output_token_ids"], cut["output_token_ids"]
    position = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if position is None and len(a) != len(b):
        position = min(len(a), len(b))
    return {
        "index": off["index"], "source_row": off.get("source_row"), "source_id": off.get("source_id"),
        "off_tokens": len(a), "cut_tokens": len(b),
        "off_finish_reason": off["finish_reason"], "cut_finish_reason": cut["finish_reason"],
        "first_different_token_index": position,
        "finish_reason_only": a == b and off["finish_reason"] != cut["finish_reason"],
        "off_token_context": a[max(0, position - 5):position + 6] if position is not None else [],
        "cut_token_context": b[max(0, position - 5):position + 6] if position is not None else [],
    }


def mismatch_examples(off_path, cut_path, off, cut, indices, limit=20):
    if not indices or not off.get("generations_file") or not cut.get("generations_file"):
        return []
    wanted = set(indices[:limit])
    result = []
    with gzip.open(off_path.parent / off["generations_file"], "rt") as a:
        with gzip.open(cut_path.parent / cut["generations_file"], "rt") as b:
            for i, (left, right) in enumerate(zip(a, b)):
                if i in wanted:
                    x, y = json.loads(left), json.loads(right)
                    if x["index"] != i or y["index"] != i or x["prompt_sha256"] != y["prompt_sha256"]:
                        raise ValueError("Saved generation rows are not aligned")
                    for row, cell in ((x, off), (y, cut)):
                        value = [row["output_token_ids"], row["finish_reason"]]
                        actual = hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
                        if actual != cell["output_hashes"][i]:
                            raise ValueError("Saved generation differs from its recorded output hash")
                    result.append(first_difference(x, y))
                if i >= max(wanted):
                    break
    if len(result) != len(wanted):
        raise ValueError("Saved generation file is incomplete")
    return result


def build_report(suite):
    metadata = validate_suite(suite)
    runs = {mode: [(suite / path, read(suite / path)) for path in load_index(suite, mode)] for mode in ("off", "cut")}
    cell_maps = {mode: [{c["benchmark"]: c for c in run["cells"]} for _, run in records] for mode, records in runs.items()}
    complete = all(len(records) == metadata["settings"]["repeats"] for records in runs.values())
    summaries, csv_rows = [], []
    for name in BENCHMARKS:
        cells = {mode: [mapping[name] for mapping in maps] for mode, maps in cell_maps.items()}
        performance = {mode: stats([c["summary"]["wall_tokens_per_second"] for c in values]) for mode, values in cells.items()}
        cross, repeat_agreement, issues = [], {}, []
        for repeat, ((off_path, off_run), (cut_path, cut_run)) in enumerate(zip(runs["off"], runs["cut"]), 1):
            off_cell, cut_cell = cells["off"][repeat - 1], cells["cut"][repeat - 1]
            checked = confidence_report.compare_reports(
                dict(off_run, cells=[off_cell]), dict(cut_run, cells=[cut_cell]), "cut"
            )[0]
            non_output_issues = [s for s in checked["issues"] if s != "output_hashes differ or are missing"]
            issues.extend(f"repeat {repeat}: {s}" for s in non_output_issues)
            agreement = parity(off_cell, cut_cell)
            cross.append({
                "repeat": repeat, **agreement, "comparison": checked,
                "mismatch_examples_first_20": mismatch_examples(
                    off_path, cut_path, off_cell, cut_cell, agreement["different_indices"]
                ),
            })
        for mode, values in cells.items():
            repeat_agreement[mode] = [{"repeat_a": 1, "repeat_b": i + 1, **parity(values[0], cell)} for i, cell in enumerate(values[1:], 1)]
            for cell in values:
                if cell["summary"]["requests"] != metadata["inventory"][name]["source_rows"]:
                    issues.append(f"{mode}: incomplete source-row coverage")
                if not cell["summary"].get("measurement_valid"):
                    issues.append(f"{mode}: invalid measurement")
                if cell["concurrency"] != metadata["settings"]["concurrency"]:
                    issues.append(f"{mode}: concurrency differs from suite settings")
        off_mean, cut_mean = performance["off"]["mean"], performance["cut"]["mean"]
        paired_complete = complete and len(cross) == metadata["settings"]["repeats"]
        compatible = paired_complete and not issues and all(c["comparable"] for c in cross)
        observed = cut_mean / off_mean if compatible else None
        strict = compatible and all(c["comparison"]["comparison_valid"] for c in cross)
        row = {
            "benchmark": name, "source_rows": metadata["inventory"][name]["source_rows"],
            "concurrency": metadata["settings"]["concurrency"],
            "off_repeats": len(cells["off"]), "cut_repeats": len(cells["cut"]),
            "off_wall_tps_mean": off_mean, "off_wall_tps_stdev": performance["off"]["stdev"],
            "cut_wall_tps_mean": cut_mean, "cut_wall_tps_stdev": performance["cut"]["stdev"],
            "workload_compatible": compatible, "observed_throughput_ratio": observed,
            "observed_percent_change": 100 * (observed - 1) if observed is not None else None,
            "cross_mode_agreement_pct_mean": statistics.mean(c["agreement_pct"] for c in cross) if cross and all(c["comparable"] for c in cross) else None,
            "off_repeat_agreement_pct_mean": mean_agreement(repeat_agreement["off"]),
            "cut_repeat_agreement_pct_mean": mean_agreement(repeat_agreement["cut"]),
            "strict_comparison_valid": strict, "strict_speedup": observed if strict else None,
            "issues": "; ".join(sorted(set(issues))),
        }
        csv_rows.append(row)
        summaries.append({**row, "throughput": performance, "cross_mode": cross, "same_mode": repeat_agreement,
                          "per_run_summaries": {mode: [c["summary"] for c in values] for mode, values in cells.items()}})
    ratios = [row["observed_throughput_ratio"] for row in csv_rows]
    overall = math.exp(statistics.mean(math.log(r) for r in ratios)) if all(r is not None and r > 0 for r in ratios) else None
    result = {
        "schema": "dspark_full_suite_report_v1", "complete": complete, "inventory": metadata["inventory"],
        "settings": metadata["settings"], "run_files": {mode: [str(path) for path, _ in records] for mode, records in runs.items()},
        "observed_geometric_mean_ratio_equal_benchmark_weight": overall,
        "strict_geometric_mean_speedup": overall if all(row["strict_comparison_valid"] for row in csv_rows) else None,
        "benchmarks": summaries,
        "notes": [
            "Observed throughput ratios are descriptive; they are not exact-output-validated speedups when output parity fails.",
            "Repeats are grouped by server mode. Standard deviations describe measured variation, not a confidence interval.",
            "Same-mode output drift is a control, not proof that all cross-mode mismatches are harmless.",
            "Full source splits, no prompt sampling; MT-Bench first turn only, LiveCodeBench release v5.",
            "Concurrency is the requested maximum; splits smaller than it cannot fill every client slot.",
            "Greedy decoding with the configured output cap. Length-limited request counts are retained per run.",
            "This is a performance/output-agreement report; task answer accuracy and generated-program pass rates are not scored.",
        ],
    }
    evaluator.write_json(suite / "suite_report.json", result)
    with (suite / "suite_report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ["# DSpark full-dataset speed comparison", "", f"Complete: {complete}. Concurrency: {metadata['settings']['concurrency']}. Proposals: 7.", "",
             "| Benchmark | Requests/pass | OFF tok/s | ON tok/s | Observed ratio | Output agreement | Strict parity valid |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in csv_rows:
        lines.append(f"| {row['benchmark']} | {row['source_rows']} | {format_number(row['off_wall_tps_mean'])} | {format_number(row['cut_wall_tps_mean'])} | {format_number(row['observed_throughput_ratio'])} | {format_number(row['cross_mode_agreement_pct_mean'], '%')} | {row['strict_comparison_valid']} |")
    lines.extend(["", *result["notes"]])
    (suite / "suite_report.md").write_text("\n".join(lines) + "\n")
    return result


def mean_agreement(rows):
    return statistics.mean(row["agreement_pct"] for row in rows) if rows and all(row["comparable"] for row in rows) else None


def format_number(value, suffix=""):
    return f"{value:.3f}{suffix}" if value is not None else "pending"


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--suite", type=Path, required=True)
    prep.add_argument("--target", type=Path, required=True)
    prep.add_argument("--draft", type=Path, required=True)
    prep.add_argument("--repeats", type=int, default=3)
    prep.add_argument("--concurrency", type=int, default=64)
    prep.add_argument("--max-new-tokens", type=int, default=2048)
    prep.add_argument("--source", action="append", default=[])
    prep.add_argument("--reuse-manifest", type=Path,
                      help="Reuse complete splits with matching model/tokenizer fingerprints; rebuild sampled splits")
    run = commands.add_parser("evaluate")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8217")
    run.add_argument("--server-manifest", type=Path, required=True)
    run.add_argument("--concurrency", type=int,
                     help="Require this concurrency in the saved suite before evaluation")
    report = commands.add_parser("report")
    report.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.repeats < 1 or args.max_new_tokens < 1 or args.concurrency < 1:
            parser.error("Use at least one repeat, a positive output-token cap and positive concurrency")
        prepare(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        build_report(args.suite.resolve())
        print(args.suite / "suite_report.csv")


if __name__ == "__main__":
    main()
