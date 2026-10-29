"""Saved-run integrity and diagnostic comparisons; no ML/runtime dependencies."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Evaluator"))
import diagnose_dspark_repeats as diagnostic


def digest(tokens):
    return hashlib.sha256(json.dumps([tokens, "stop"], separators=(",", ":")).encode()).hexdigest()


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original, self.rerun = self.root / "original", self.root / "rerun"
        for suite, mode in ((self.original, "off"), (self.original, "cut"), (self.rerun, "cut")):
            self.make_runs(suite, mode)

    def make_runs(self, suite, mode):
        paths = []
        for repeat in (1, 2, 3):
            directory = suite / "runs" / f"{mode}_r{repeat}"
            directory.mkdir(parents=True)
            tokens = [[1, 2], [3, 4, 5]]
            rows = [{"index": i, "prompt_sha256": f"prompt-{i}", "output_token_ids": ids,
                     "completion_tokens": len(ids), "finish_reason": "stop", "latency_seconds": 2.0}
                    for i, ids in enumerate(tokens)]
            with gzip.open(directory / "generations.gz", "wt") as target:
                for row in rows:
                    target.write(json.dumps(row) + "\n")
            wall = float(repeat + 1)
            cell = {"benchmark": "example", "temperature": 0.0, "concurrency": 128,
                    "sampling": {"temperature": 0}, "prompt_hashes": [r["prompt_sha256"] for r in rows],
                    "output_hashes": [digest(ids) for ids in tokens], "generations_file": "generations.gz",
                    "summary": {"requests": 2, "output_tokens": 5, "wall_seconds": wall,
                                "summed_request_seconds": 4, "wall_tokens_per_second": 5 / wall,
                                "tokens_per_second": 5 / 4, "length_limited_requests": 0,
                                "acceptance_length_with_bonus_pooled": 2, "draft_acceptance_rate": 0.5,
                                "measurement_valid": True}}
            server = {"target_sha256": "target", "draft_sha256": "draft", "source_sha256": {"source": "hash"},
                      "block_size": 8, "proposals": 7, "max_num_seqs": 128, "max_batched_tokens": 5120,
                      "memory_utilization": 0.5, "seed": 42, "kv_block_size": 128,
                      "target_execution_requested": "eager", "draft_execution_requested": "eager",
                      "dspark_head_graph": False, "dspark_confidence": {"mode": mode, "trace_directory": None}}
            run = {"complete": True, "manifest_sha256": "manifest", "eval_code_sha256": "code",
                   "comparison_settings": {"runtime_identity": {"additional_config": {
                       "dspark_confidence_config": {"mode": mode, "trace_directory": None, "threshold": 0.5}}}},
                   "server_manifest": server, "cells": [cell]}
            (directory / "run_info.json").write_text(json.dumps(run))
            paths.append(str((directory / "run_info.json").relative_to(suite)))
        (suite / f"runs_{mode}.json").write_text(json.dumps(paths))

    def group(self, mode="off"):
        return diagnostic.load_group(self.original, mode, mode)

    def test_real_counters_compute_overlap_and_repeat_variation(self):
        result = diagnostic.build({"off": self.group(), "on": self.group("cut")})
        self.assertEqual([r["average_client_in_flight"] for r in result["per_run"][:3]], [2.0, 4 / 3, 1.0])
        self.assertGreater(result["groups"][0]["throughput"]["cv_pct"], 0)
        self.assertEqual(result["comparisons"][0]["strict_ratio_of_means"], 1)
        self.assertEqual(len(result["comparisons"][0]["all_repeat_pairs"]), 9)

    def test_same_mode_output_drift_is_not_hidden(self):
        group = self.group()
        group[1]["cells"][("example", 0.0, 128)]["cell"]["output_hashes"][0] = "changed"
        result = diagnostic.build({"off": group})
        agreement = result["groups"][0]["same_mode_output_agreement_pct"]
        self.assertEqual(agreement["min"], 50)
        self.assertEqual(agreement["max"], 100)

    def test_corrupted_generation_is_rejected(self):
        path = self.original / "runs/off_r1/generations.gz"
        with gzip.open(path, "rt") as source:
            rows = [json.loads(line) for line in source]
        rows[0]["output_token_ids"][0] = 999
        with gzip.open(path, "wt") as target:
            for row in rows:
                target.write(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "content/hash mismatch"):
            self.group()

    def test_different_prompts_or_code_prevent_strict_comparison(self):
        a, b = self.group()[0], self.group("cut")[0]
        key = ("example", 0.0, 128)
        altered = copy.deepcopy(b)
        altered["report"]["eval_code_sha256"] = "other-code"
        result = diagnostic.compare(a, altered, key)
        self.assertFalse(result["strict_output_comparison_valid"])
        self.assertIsNone(result["observed_wall_tps_ratio"])
        altered = copy.deepcopy(b)
        altered["cells"][key]["cell"]["prompt_hashes"].reverse()
        result = diagnostic.compare(a, altered, key)
        self.assertIsNone(result["output_agreement_pct"])
        self.assertIn("Ordered prompts differ", result["issues"])

    def test_cli_preserves_sources_and_refuses_overwrite(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = self.root / "diagnostic"
        command = [sys.executable, str(ROOT / "Evaluator/diagnose_dspark_repeats.py"),
                   "--original-suite", str(self.original), "--rerun-suite", str(self.rerun), "--output", str(output)]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "diagnostic.json").read_text())
        self.assertEqual(len(report["per_run"]), 9)
        self.assertEqual(len(report["comparisons"]), 3)
        self.assertTrue((output / "per_run.csv").exists())
        self.assertEqual(before, {p: Path(p).read_bytes() for p in before})
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
