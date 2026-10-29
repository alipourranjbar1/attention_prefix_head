#!/usr/bin/env python3
"""Preflight, server ownership checks and final parity gate for ON then OFF."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys


EXPECTED_ENV = {
    "VLLM_BATCH_INVARIANT": "1", "HCCL_DETERMINISTIC": "strict", "LCCL_DETERMINISTIC": "1",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0", "ATB_LLM_LCOC_ENABLE": "0", "PYTHONHASHSEED": "42",
}


def read(path):
    return json.loads(path.read_text())


def preflight(root, output):
    for name, value in EXPECTED_ENV.items():
        if os.environ.get(name) != value:
            raise ValueError(f"Expected {name}={value}; actual value is {os.environ.get(name)!r}")
    import torch
    import torch_npu
    import vllm
    import vllm_ascend
    from vllm_ascend import batch_invariant

    for module, directory in ((vllm, "vllm"), (vllm_ascend, "vllm-ascend")):
        if not Path(module.__file__).resolve().is_relative_to((root / directory).resolve()):
            raise ValueError(f"Loaded the wrong repository: {module.__file__}")
    if not batch_invariant.HAS_ASCENDC_BATCH_INVARIANT:
        raise ValueError(
            "AscendC batch_invariant_ops is missing. A Triton-only fallback does not install the "
            "batch-invariant attention replacement, so this experiment stops here. "
            f"Install the matching 910B operators using {root / 'vllm-ascend/csrc/build_batch_invariant_ops.sh'} "
            "ascend910b in your activated CANN/Conda environment, then rerun. "
            "No benchmark was started and no results were overwritten."
        )
    required = (
        "npu_mm_batch_invariant", "npu_matmul_batch_invariant",
        "npu_fused_infer_attention_score_batch_invariant", "npu_reduce_sum_batch_invariant",
    )
    for name in required:
        if not hasattr(torch.ops.batch_invariant_ops, name):
            raise ValueError(f"Installed batch_invariant_ops lacks {name}; install operators matching this checkout")
    payload = {
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "vllm_source": vllm.__file__, "vllm_ascend_source": vllm_ascend.__file__,
        "ascendc_batch_invariant": True, "triton": bool(batch_invariant.HAS_TRITON),
        "registered_operator_names": list(required), "environment": EXPECTED_ENV,
        "note": "Operator availability only. Actual Qwen3/DSpark output parity must pass after inference.",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print("Batch-invariant runtime/operator preflight passed.")


def ready(manifest, pid, mode, suite):
    report = read(manifest)
    if report.get("launcher_pid") != pid:
        raise ValueError("The ready endpoint belongs to a different launcher; refusing to evaluate it")
    if os.getpgid(pid) != pid:
        raise ValueError("Server is not in its expected private process group")
    if not manifest.resolve().is_relative_to(suite.resolve()):
        raise ValueError("Server manifest does not belong to this result directory")
    confidence = report.get("dspark_confidence") or {}
    if confidence.get("mode") != mode or confidence.get("trace_directory"):
        raise ValueError("Wrong confidence mode or diagnostic tracing enabled")
    if report.get("max_num_seqs") != 128 or report.get("proposals") != 7 or report.get("seed") != 42:
        raise ValueError("Server must use concurrency capacity 128, seven proposals and seed 42")
    if report.get("determinism_environment") != EXPECTED_ENV:
        raise ValueError("Server did not inherit all requested determinism settings")
    spec = report.get("speculative_config") or {}
    if spec.get("draft_sample_method") != "greedy" or spec.get("rejection_sample_method") != "standard":
        raise ValueError("Expected greedy drafts and standard target verification")
    log = Path(report["server_log"]).read_text(errors="replace")
    if "Enabling batch-invariant mode for vLLM on Ascend NPU." not in log:
        raise ValueError("Server is ready but did not confirm batch-invariant initialization; see server.log")
    print(f"Verified owned {mode} server and batch-invariant startup.")


def finish(suite):
    report = read(suite / "suite_report.json")
    if not report.get("complete") or report["settings"]["repeats"] != 1 or report["settings"]["concurrency"] != 128:
        raise ValueError("Expected exactly one complete ON and OFF run at concurrency 128")
    for mode, label in (("cut", "ON"), ("off", "OFF")):
        paths = read(suite / f"runs_{mode}.json")
        if len(paths) != 1:
            raise ValueError(f"Expected one indexed {label} result")
        (suite / label).symlink_to(Path(paths[0]).parent, target_is_directory=True)
    print("\nONE-PASS ON / OFF COMPARISON (wall-clock throughput)")
    print("dataset OFF_tok/s ON_tok/s observed_ratio exact_output_agreement strict_valid")
    for row in report["benchmarks"]:
        ratio = row["observed_throughput_ratio"]
        agreement = row["cross_mode_agreement_pct_mean"]
        print(f"{row['benchmark']} {row['off_wall_tps_mean']:.2f} {row['cut_wall_tps_mean']:.2f} "
              f"{ratio if ratio is not None else 'NA'} {agreement if agreement is not None else 'NA'} "
              f"{row['strict_comparison_valid']}")
    valid = all(row["strict_comparison_valid"] for row in report["benchmarks"])
    status = {
        "complete": True, "order": ["cut", "off"], "runs_per_mode": 1, "concurrency": 128,
        "strict_output_comparison_valid": valid,
        "note": "Single measurements: no repeatability estimate or statistical significance claim. "
                "Batch-invariant performance is a separate configuration from previous default-kernel runs.",
    }
    (suite / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"\nON results: {suite / 'ON/summary.csv'}")
    print(f"OFF results: {suite / 'OFF/summary.csv'}")
    print(f"Comparison: {suite / 'suite_report.csv'}")
    print(f"Outputs/mismatch details: {suite / 'suite_report.json'}")
    print(status["note"])
    if not valid:
        print("OUTPUT PARITY OR CONFIGURATION CHECK FAILED. Both results are saved; no validated speedup is claimed.")
        return 2
    print("Exact output parity passed for every benchmark in this ON/OFF pair.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    port = commands.add_parser("port-free")
    port.add_argument("--port", type=int, required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--root", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("ready")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--pid", type=int, required=True)
    check.add_argument("--mode", choices=("cut", "off"), required=True)
    check.add_argument("--suite", type=Path, required=True)
    end = commands.add_parser("finish")
    end.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "port-free":
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", args.port))
            except OSError as exc:
                raise ValueError(f"Port {args.port} is busy. Stop the existing evaluation server first; no process was killed.") from exc
    elif args.command == "preflight":
        preflight(args.root, args.output)
    elif args.command == "ready":
        ready(args.manifest, args.pid, args.mode, args.suite)
    else:
        return finish(args.suite)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, ImportError) as error:
        sys.exit(str(error))
