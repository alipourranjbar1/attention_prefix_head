#!/usr/bin/env python3
"""Validate saved results from a real Ascend ACTION=all confidence experiment."""

import argparse
import json
import math
from pathlib import Path


def check(directory):
    directory = Path(directory)
    accuracy = json.loads((directory / "confidence_accuracy.json").read_text())
    cut_accuracy = json.loads((directory / "confidence_accuracy_after_cut.json").read_text())
    comparison = json.loads((directory / "speed_comparison.json").read_text())
    if not accuracy["cells"] or not comparison["cells"]:
        raise AssertionError("Missing measured cells")
    for cell in accuracy["cells"]:
        metrics = cell["full_block_accuracy"]
        assert metrics["rounds"] > 0, "No complete measured confidence labels"
        assert math.isclose(sum(metrics[k] for k in ("exact_pct", "under_pct", "over_pct")), 100), metrics
    assert cut_accuracy["cells"], "Missing cut-mode accuracy"
    for cell in cut_accuracy["cells"]:
        assert cell["modes"] == ["cut"], cell
        observed = cell["observed_after_verification"]
        assert observed["rounds"] > 0, "No measured cut-mode confidence labels"
        metrics = observed["capped_prediction_accuracy"]
        assert math.isclose(sum(metrics[k] for k in ("exact_pct", "under_pct", "over_pct")), 100), metrics
    assert {cell["mode"] for cell in comparison["cells"]} == {"observe", "cut"}
    for cell in comparison["cells"]:
        assert cell["comparison_valid"], cell["issues"]
        assert cell["speedup"] > 0, cell
    off = json.loads((directory / "eval_off/run_info.json").read_text())
    observe = json.loads((directory / "eval_observe/run_info.json").read_text())
    lookup = {(c["benchmark"], c["temperature"], c.get("concurrency", 1)): c for c in off["cells"]}
    for cell in observe["cells"]:
        key = (cell["benchmark"], cell["temperature"], cell.get("concurrency", 1))
        # At c=1, head observation must leave full proposals/rejection counts
        # unchanged. At c>1, batch composition can differ between timed runs;
        # output parity remains mandatory in the comparison report.
        if key[2] == 1:
            assert cell["speculative_hashes"] == lookup[key]["speculative_hashes"], (
                "Head observation changed c=1 proposals"
            )
    return "PASS: measured labels, matched outputs/timing, and c=1 proposal parity"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("experiment", type=Path)
    print(check(parser.parse_args().experiment))
