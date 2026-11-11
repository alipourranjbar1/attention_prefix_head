"""Read-only allocation/dependency checks. Never stops other processes."""
import argparse
import json
from pathlib import Path
import re
import subprocess

PINS = {"vllm": "77b1d88c5b52abaf6a103292119f3435b51699d9",
        "vllm-ascend": "276a5c840b21c83b153d3ebd831720bc926dfc4e"}


def devices(targets, trainers):
    groups = [s.split(",") for s in (targets, trainers)]
    all_ids = sum(groups, [])
    if not all(x.isdigit() and 0 <= int(x) < 8 for x in all_ids) or len(set(all_ids)) != len(all_ids):
        raise ValueError("Use distinct numeric target/trainer NPU IDs in 0..7")
    return all_ids


def require_free(table, ids):
    for device in ids:
        if not re.search(r"No running processes found in NPU\s+" + re.escape(device) + r"\s*\|", table):
            raise ValueError(f"NPU {device} is occupied or its status is unclear. Nothing was stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependencies", required=True)
    parser.add_argument("--speculators", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--trainers", required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    ids = devices(args.targets, args.trainers)
    for name, expected in PINS.items():
        path = Path(args.dependencies) / name
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError(f"{name}: expected {expected}, found {actual}; existing checkout was not changed")
    spec = Path(args.speculators)
    for relative in ("scripts/launch_vllm.py", "src/speculators/models/dspark/model_definitions.py"):
        if not (spec / relative).is_file():
            raise ValueError(f"Missing existing dependency: {spec / relative}")
    if not args.metadata_only:
        table = subprocess.check_output(["npu-smi", "info"], text=True)
        require_free(table, ids)
    print(json.dumps({"preflight": "passed", "allocated_npus": ids, "npu_status_checked": not args.metadata_only}))


if __name__ == "__main__":
    main()
