"""Full coverage, repeat controls, output retention and suite orchestration tests."""

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Evaluator"))
import dspark_benchmark_suite as suite
import evaluator


def output_hash(tokens, reason="stop"):
    return hashlib.sha256(json.dumps([tokens, reason], separators=(",", ":")).encode()).hexdigest()


class FullSuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def test_full_preparation_exceeds_old_sample_and_adjusts_context(self):
        target, draft = self.path / "target", self.path / "draft"
        target.mkdir()
        draft.mkdir()
        (target / "config.json").write_text('{"max_position_embeddings": 8192}')
        (draft / "config.json").write_text('{}')
        args = argparse.Namespace(suite=self.path / "suite", target=target, draft=draft,
                                  repeats=3, max_new_tokens=2048, source=[])
        def load_rows(name, *_):
            count = 130 if name == "gsm8k" else 2
            return [dict(question="q", problem="p", prompt="p", instruction="i", input="",
                         turns=["t"], question_content="q", id=i) for i in range(count)], {"test": name}, False
        tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: [1] * 6000)
        transformer = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer))
        with patch.dict(sys.modules, {"transformers": transformer}), patch.object(evaluator, "load_rows", side_effect=load_rows):
            suite.prepare(args)
        saved = suite.read(args.suite / "suite.json")
        manifest = suite.read(args.suite / "prompts.json")
        self.assertEqual(saved["inventory"]["gsm8k"]["requests"], 130)
        self.assertEqual(saved["max_model_len"], 8064)
        self.assertEqual(list(manifest["datasets"]), list(suite.BENCHMARKS))
        evaluator.require_full_manifest(manifest, suite.BENCHMARKS)
        # Reuse the frozen file without reloading/tokenizing sources.
        with patch.object(evaluator, "prepare_manifest", side_effect=AssertionError("Must reuse")):
            suite.prepare(args)
        manifest["datasets"]["gsm8k"]["results"].pop()
        with self.assertRaisesRegex(ValueError, "all source rows"):
            evaluator.require_full_manifest(manifest, suite.BENCHMARKS)

    def test_full_manifest_rejects_duplicate_source_rows(self):
        manifest = {"datasets": {"gsm8k": {"source_rows": 2, "results": [{"source_row": 0}] * 2}}}
        with self.assertRaisesRegex(ValueError, "duplicated"):
            evaluator.require_full_manifest(manifest, ["gsm8k"])

    def reuse_fixture(self):
        target, draft = self.path / "target", self.path / "draft"
        target.mkdir()
        draft.mkdir()
        (target / "config.json").write_text('{"max_position_embeddings": 8192}')
        (target / "tokenizer.json").write_text('fixture tokenizer')
        (draft / "config.json").write_text('{}')
        manifest = {
            "format": 3, "thinking": False,
            "target_config_sha256": evaluator.file_hash(target / "config.json"),
            "tokenizer_sha256": {"tokenizer.json": evaluator.file_hash(target / "tokenizer.json")},
            "choices": {"lcb_release": "release_v5", "formatter_sha256": "saved_formatter"},
            "datasets": {},
        }
        for name in evaluator.COUNTS:
            count = 1 if name == "alpaca" else 2
            rows = [{"source_row": i, "source_id": str(i), "prompt_token_ids": [7, i],
                     "sha256": evaluator.digest([7, i])} for i in range(count)]
            manifest["datasets"][name] = {"source_rows": 2, "requests": count, "results": rows,
                                         "source": {"fixture": name}}
        source = self.path / "saved_prompts.json"
        evaluator.write_json(source, manifest)
        args = argparse.Namespace(suite=self.path / "suite", target=target, draft=draft,
                                  repeats=3, max_new_tokens=2048, source=[], reuse_manifest=source)
        return args, manifest

    def test_reuses_all_seven_without_raw_cache_and_excludes_alpaca(self):
        args, original = self.reuse_fixture()
        tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: [1, 2, 3])
        transformer = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer))
        with patch.dict(sys.modules, {"transformers": transformer}), patch.object(evaluator, "load_rows", side_effect=AssertionError("Must not fetch datasets")):
            suite.prepare(args)
        result = suite.read(args.suite / "prompts.json")
        self.assertEqual(result["datasets"]["lcb"], original["datasets"]["lcb"])
        self.assertNotIn("alpaca", result["datasets"])
        self.assertNotIn("alpaca", suite.read(args.suite / "suite.json")["inventory"])
        self.assertEqual(result["prepared_manifest_reuse"]["sha256"], evaluator.file_hash(args.reuse_manifest))
        self.assertEqual(len(result["prepared_manifest_reuse"]["datasets"]), 7)
        evaluator.require_full_manifest(result, suite.BENCHMARKS)
        evaluator.select_prompts(result, suite.BENCHMARKS, 0, False)

    def test_concurrency_128_preparation_preserves_64_suite_and_prompts(self):
        args, _ = self.reuse_fixture()
        tokenizer = SimpleNamespace(apply_chat_template=lambda *_a, **_k: [1, 2, 3])
        transformer = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer))
        saved = {}
        with patch.dict(sys.modules, {"transformers": transformer}), patch.object(evaluator, "load_rows", side_effect=AssertionError("Must reuse")):
            for concurrency in (64, 128):
                args.suite = self.path / f"suite_c{concurrency}"
                args.concurrency = concurrency
                suite.prepare(args)
                saved[concurrency] = suite.read(args.suite / "suite.json")
                self.assertEqual(saved[concurrency]["settings"]["concurrency"], concurrency)
                if concurrency == 64:
                    baseline = (args.suite / "suite.json").read_bytes()
        self.assertEqual(saved[64]["manifest_sha256"], saved[128]["manifest_sha256"])
        self.assertEqual(saved[64]["max_model_len"], saved[128]["max_model_len"])
        self.assertEqual((self.path / "suite_c64/suite.json").read_bytes(), baseline)
        args.suite = self.path / "suite_c64"
        with self.assertRaisesRegex(ValueError, "Existing suite settings"):
            suite.prepare(args)

    def test_reuse_checks_fingerprints_release_token_integrity_and_coverage(self):
        args, original = self.reuse_fixture()
        for change, message in (
            (lambda m: m.update(tokenizer_sha256={}), "fingerprint"),
            (lambda m: m.update(target_config_sha256="other"), "fingerprint"),
            (lambda m: m.update(thinking=True), "fingerprint"),
            (lambda m: m["choices"].update(lcb_release="release_v6"), "release must be v5"),
            (lambda m: m["datasets"]["lcb"]["results"][0].update(sha256="wrong"), "token hash"),
            (lambda m: m["datasets"]["lcb"]["results"][1].update(source_row=0), "duplicated"),
        ):
            with self.subTest(message=message):
                changed = json.loads(json.dumps(original))
                change(changed)
                evaluator.write_json(args.reuse_manifest, changed)
                with self.assertRaisesRegex(ValueError, message):
                    suite.reusable_datasets(args.reuse_manifest, args.target, [])

    def test_reuse_respects_explicit_sources_and_does_not_reuse_sampled_lcb(self):
        args, saved = self.reuse_fixture()
        saved["datasets"]["lcb"]["results"].pop()
        evaluator.write_json(args.reuse_manifest, saved)
        reused, _ = suite.reusable_datasets(args.reuse_manifest, args.target, ["gsm8k=/local/full.jsonl"])
        self.assertNotIn("gsm8k", reused)
        self.assertNotIn("lcb", reused)
        self.assertNotIn("alpaca", reused)
        self.assertIn("humaneval", reused)

    def prepare_report_fixture(self, concurrency=64):
        (self.path / "prompts.json").write_text('{}')
        metadata = {
            "settings": {"repeats": 2, "target": "/fixture/target", "draft": "/fixture/draft",
                         "max_new_tokens": 2048, "concurrency": concurrency,
                         "evaluator_sha256": evaluator.file_hash(ROOT / "Evaluator/evaluator.py")},
            "manifest_sha256": evaluator.file_hash(self.path / "prompts.json"),
            "inventory": {name: {"requests": 2, "source_rows": 2} for name in suite.BENCHMARKS},
            "max_model_len": 5120, "max_batched_tokens": 5120,
        }
        evaluator.write_json(self.path / "suite.json", metadata)
        return metadata

    def write_run(self, mode, repeat, changed=False, output=None):
        concurrency = suite.read(self.path / "suite.json")["settings"]["concurrency"]
        output = output or self.path / f"{mode}_{repeat}"
        output.mkdir(parents=True)
        server = {key: "same" for key in (
            "target_sha256", "draft_sha256", "source_sha256", "block_size", "proposals", "max_num_seqs",
            "max_batched_tokens", "memory_utilization", "seed", "kv_block_size", "target_execution_requested",
            "draft_execution_requested", "dspark_head_graph",
        )}
        server["dspark_confidence"] = {"mode": mode, "trace_directory": None}
        server["max_num_seqs"] = concurrency
        report = {"complete": True, "manifest_sha256": "frozen", "eval_code_sha256": "same",
                  "server_manifest": server, "comparison_settings": {"same": True}, "cells": []}
        for name in suite.BENCHMARKS:
            rows = [{"source_row": i, "source_id": str(i), "prompt_sha256": f"p{i}",
                     "output_token_ids": [10, 99 if changed and name == "gsm8k" and i == 0 else 11],
                     "finish_reason": "stop", "text": "output", "completion_tokens": 2,
                     "request_id": f"r{i}", "latency_seconds": 1} for i in range(2)]
            generation = evaluator.save_generations(output, name, 0, concurrency, rows)
            cell = {"benchmark": name, "temperature": 0, "concurrency": concurrency, "sampling": {"temperature": 0},
                    "prompt_hashes": [r["prompt_sha256"] for r in rows],
                    "output_hashes": [output_hash(r["output_token_ids"]) for r in rows],
                    "generations_file": generation,
                    "summary": {"requests": 2, "measurement_valid": True, "tokens_per_second": 40,
                                "wall_tokens_per_second": 1000 + repeat * 200 + (100 if mode == "cut" else 0),
                                "acceptance_length_with_bonus_pooled": 6, "output_tokens": 4}}
            report["cells"].append(cell)
        evaluator.write_json(output / "run_info.json", report)
        return output / "run_info.json"

    def add_runs(self, changed=False, concurrency=64):
        self.prepare_report_fixture(concurrency)
        for mode in ("off", "cut"):
            paths = [self.write_run(mode, i, changed=changed and mode == "cut") for i in range(2)]
            evaluator.write_json(self.path / f"runs_{mode}.json", [str(p.relative_to(self.path)) for p in paths])

    def test_output_mismatch_remains_unvalidated_with_full_dataset(self):
        self.add_runs(changed=True)
        report = suite.build_report(self.path)
        row = report["benchmarks"][0]
        self.assertTrue(report["complete"])
        self.assertTrue(row["workload_compatible"])
        self.assertAlmostEqual(row["observed_throughput_ratio"], 1200 / 1100)
        self.assertIsNone(row["strict_speedup"])
        self.assertFalse(row["strict_comparison_valid"])
        self.assertEqual(row["cross_mode_agreement_pct_mean"], 50)
        self.assertEqual(row["off_repeat_agreement_pct_mean"], 100)
        self.assertEqual(row["cut_repeat_agreement_pct_mean"], 100)
        example = row["cross_mode"][0]["mismatch_examples_first_20"][0]
        self.assertEqual(example["first_different_token_index"], 1)
        self.assertIsNone(report["strict_geometric_mean_speedup"])
        self.assertTrue((self.path / "suite_report.md").exists())
        with gzip.open(self.path / "cut_0/generations/gsm8k_t0_c64.jsonl.gz", "rt") as stream:
            self.assertEqual(json.loads(stream.readline())["output_token_ids"], [10, 99])

    def test_matching_outputs_allow_strict_report(self):
        self.add_runs()
        report = suite.build_report(self.path)
        self.assertAlmostEqual(report["strict_geometric_mean_speedup"], 1200 / 1100)
        self.assertEqual(report["benchmarks"][0]["cross_mode_agreement_pct_mean"], 100)

    def test_concurrency_128_report_uses_saved_concurrency(self):
        self.add_runs(concurrency=128)
        report = suite.build_report(self.path)
        self.assertEqual(report["benchmarks"][0]["concurrency"], 128)
        self.assertTrue(report["benchmarks"][0]["strict_comparison_valid"])
        self.assertIn("Concurrency: 128", (self.path / "suite_report.md").read_text())
        # Even matching OFF/CUT files at another concurrency must not be
        # presented as this suite's result.
        metadata = suite.read(self.path / "suite.json")
        metadata["settings"]["concurrency"] = 64
        evaluator.write_json(self.path / "suite.json", metadata)
        report = suite.build_report(self.path)
        self.assertIsNone(report["benchmarks"][0]["observed_throughput_ratio"])
        self.assertIn("concurrency differs", report["benchmarks"][0]["issues"])

    def test_runtime_mismatch_prevents_even_observed_ratio(self):
        self.add_runs()
        path = self.path / "cut_1/run_info.json"
        report = suite.read(path)
        report["server_manifest"]["source_sha256"] = "changed"
        evaluator.write_json(path, report)
        result = suite.build_report(self.path)
        self.assertIsNone(result["benchmarks"][0]["observed_throughput_ratio"])
        self.assertIn("source_sha256", result["benchmarks"][0]["issues"])

    def test_evaluate_runs_all_datasets_repeats_and_resumes(self):
        self.exercise_evaluation(64)

    def test_evaluate_at_128_and_reject_client_or_server_mismatch(self):
        args, launch = self.exercise_evaluation(128)
        with patch.object(suite.subprocess, "run", side_effect=AssertionError("Must not evaluate mismatched config")):
            args.concurrency = 64
            with self.assertRaisesRegex(ValueError, "Requested concurrency"):
                suite.evaluate(args)
            args.concurrency = 128
            launch["max_num_seqs"] = 64
            evaluator.write_json(args.server_manifest, launch)
            with self.assertRaisesRegex(ValueError, "Server does not match"):
                suite.evaluate(args)

    def exercise_evaluation(self, concurrency):
        self.prepare_report_fixture(concurrency)
        launch = {"kind": "dspark", "proposals": 7, "max_num_seqs": concurrency, "max_model_len": 5120,
                  "max_batched_tokens": 5120, "target": "/fixture/target", "draft": "/fixture/draft",
                  "dspark_confidence": {"mode": "off", "trace_directory": None}}
        evaluator.write_json(self.path / "launch.json", launch)
        args = argparse.Namespace(suite=self.path, server_manifest=self.path / "launch.json", base_url="http://fake", concurrency=concurrency)
        commands = []
        def evaluate_fake(command, check):
            self.assertTrue(check)
            self.assertIn("--all-prompts", command)
            self.assertIn("--save-generations", command)
            start = command.index("--datasets") + 1
            self.assertEqual(len(suite.BENCHMARKS), 7)
            self.assertNotIn("alpaca", command)
            self.assertEqual(command[start:start + len(suite.BENCHMARKS)], list(suite.BENCHMARKS))
            self.assertEqual(command[command.index("--concurrencies") + 1], str(concurrency))
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            self.write_run("off", len(commands) - 1, output=output)
        with patch.object(suite.subprocess, "run", side_effect=evaluate_fake):
            suite.evaluate(args)
            suite.evaluate(args)
        self.assertEqual(len(commands), 2)
        self.assertEqual(len(suite.load_index(self.path, "off")), 2)
        self.assertFalse(suite.read(self.path / "suite_report.json")["complete"])
        return args, launch


if __name__ == "__main__":
    unittest.main()
