"""CPU contract tests; run directly without the NPU plugin conftest."""

import ast
import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import patch

import torch
from pydantic import ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[4]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = load("dspark_confidence_runtime", "vllm-ascend/vllm_ascend/spec_decode/dspark_confidence.py")
reporter = load("dspark_confidence_report", "Evaluator/dspark_confidence_report.py")
launcher = load("dspark_confidence_launcher", "speculators/scripts/dflash_prefix_compiled/target_graph_server.py")
config_utils = load("dspark_config_utils", "vllm-ascend/vllm_ascend/config_utils.py")


def extract_class(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), relative, "exec"), namespace)
    return namespace[name]


Scheduler = extract_class(
    "vllm-ascend/vllm_ascend/spec_decode/utils.py", "DynamicSpecScheduler", dict(torch=torch, math=math, Any=Any)
)
Config = extract_class(
    "vllm-ascend/vllm_ascend/ascend_config.py",
    "DSparkConfidenceConfig",
    dict(config=config_utils.config, model_validator=model_validator, math=math, Literal=Literal),
)
META = SimpleNamespace(all_greedy=True, no_penalties=True, allowed_token_ids_mask=None, bad_words_token_ids=None)


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def controller(self, mode="observe", trace=True, k=3, budget=2):
        scheduler = Scheduler(
            method="dspark",
            method_params=dict(initial_verify_budget_per_req=budget, budget_update_interval=1000),
            max_batch_size=128,
            num_speculative_tokens=k,
            device=torch.device("cpu"),
        )
        result = runtime.DSparkConfidence(
            mode=mode,
            scheduler=scheduler,
            max_batch_size=128,
            num_proposals=k,
            trace_directory=self.directory if trace else None,
            eos_token_ids=[99],
        )
        if result.trace:
            self.addCleanup(result.trace.close)
        return result

    def predict(self, controller, probabilities, request_ids=None):
        probabilities = torch.tensor(probabilities, dtype=torch.float32)
        batch, k = probabilities.shape
        tokens = torch.arange(10, 10 + k + 1)[None].expand(batch, -1).clone()
        model = SimpleNamespace(
            model=SimpleNamespace(confidence_head=object()),
            markov_embed=lambda ids: ids[..., None].float(),
            compute_confidence=lambda hidden, embed: probabilities.flatten(),
        )
        controller.predict(model, torch.zeros(batch * k, 4), tokens, META)
        controller.queue_copy(request_ids or [f"r{i}" for i in range(batch)])
        return tokens[:, 1:].tolist()

    def rows(self):
        return [
            json.loads(line) for path in self.directory.glob("rounds_*.jsonl") for line in path.read_text().splitlines()
        ]

    def test_observe_keeps_full_proposals_cut_follows_request_identity(self):
        for mode in ("observe", "cut"):
            controller = self.controller(mode)
            tokens = self.predict(controller, [[1, 1, 1], [0, 0, 0]], ["a", "b"])
            output = controller.submit(["b", "a"], [tokens[1], tokens[0]])
            self.assertEqual(output, [tokens[1][:1], tokens[0]] if mode == "cut" else [tokens[1], tokens[0]])
            self.assertEqual(controller.awaiting["b"]["predicted_accepted"], 0)
            self.assertEqual(controller.awaiting["a"]["predicted_accepted"], 3)
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                controller.submit(["a"], [tokens[0]])

    def test_exact_rejection_and_all_accept_have_correct_bonus_convention(self):
        c = self.controller()
        for accepted in range(4):
            tokens = self.predict(c, [[0.9, 0.9, 0.1]])[0]
            c.submit(["r0"], [tokens])
            c.verified(["r0"], {"r0": tokens}, [tokens[:accepted] + [77]])
            row = self.rows()[-1]
            self.assertEqual(row["actual_accepted_full"], accepted)
            self.assertEqual(row["actual_acceptance_length_full"], accepted + 1)
            self.assertFalse(row["censored"])
        metrics = reporter.accuracy(self.rows())
        self.assertEqual(metrics["full_block_accuracy"]["rounds"], 4)
        self.assertEqual(metrics["before_cut"]["mean_actual_acceptance_length"], 2.5)

    def test_shortened_blocks_do_not_invent_full_length_labels(self):
        c = self.controller("cut", k=4)
        tokens = self.predict(c, [[1, 1, 1, 1]])[0]
        submitted = c.submit(["r0"], [tokens])[0]
        self.assertEqual(len(submitted), 2)
        c.verified(["r0"], {"r0": submitted}, [submitted + [tokens[2]]])
        row = self.rows()[-1]
        self.assertTrue(row["censored"])
        self.assertIsNone(row["actual_accepted_full"])
        self.assertEqual(row["accepted_full_lower_bound"], 3)
        self.assertEqual(row["acceptance_length_after_cut"], 3)
        metrics = reporter.accuracy([row])
        self.assertEqual(metrics["full_block_accuracy"]["rounds"], 0)
        self.assertEqual(metrics["observed_after_verification"]["full_length_censored_rounds"], 1)

    def test_bonus_matching_last_proposal_completes_the_full_boundary(self):
        c = self.controller("cut")
        tokens = self.predict(c, [[1, 1, 1]])[0]
        submitted = c.submit(["r0"], [tokens])[0]
        c.verified(["r0"], {"r0": submitted}, [tokens])
        row = self.rows()[-1]
        self.assertFalse(row["censored"])
        self.assertEqual(row["actual_accepted_full"], 3)
        self.assertEqual(row["accepted_after_cut"], 2)

    def test_bonus_mismatch_after_shortened_block_exposes_exact_boundary(self):
        c = self.controller("cut")
        tokens = self.predict(c, [[1, 1, 1]])[0]
        submitted = c.submit(["r0"], [tokens])[0]
        c.verified(["r0"], {"r0": submitted}, [submitted + [77]])
        row = self.rows()[-1]
        self.assertFalse(row["censored"])
        self.assertEqual(row["actual_accepted_full"], 2)
        self.assertFalse(row["full_block_verified"])

    def test_scheduler_clipping_eos_and_discard_are_not_full_labels(self):
        c = self.controller()
        for output, discarded, reason in (
            ([11, 99], (), "eos_round"),
            ([11, 77], (0,), "discarded_or_no_verification"),
        ):
            tokens = self.predict(c, [[1, 1, 1]])[0]
            c.submit(["r0"], [tokens])
            c.verified(["r0"], {"r0": tokens[:1]}, [output], discarded)
            self.assertEqual(self.rows()[-1]["reason"], reason)
        tokens = self.predict(c, [[1, 1, 1]])[0]
        c.submit(["r0"], [tokens])
        c.verified(["r0"], {"r0": tokens[:1]}, [[11, 12]])
        self.assertTrue(self.rows()[-1]["scheduler_clipped"])
        self.assertTrue(self.rows()[-1]["censored"])

    def test_invalid_predictions_and_stale_tokens_fail(self):
        c = self.controller(trace=False)
        tokens = self.predict(c, [[float("nan"), 0.9, 0.8]])
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            c.submit(["r0"], tokens)
        tokens = self.predict(c, [[0.9, 0.9, 0.8]])
        with self.assertRaisesRegex(RuntimeError, "differ"):
            c.submit(["r0"], [[1, 2, 3]])
        self.predict(c, [[0.9, 0.9, 0.8]])
        with self.assertRaisesRegex(RuntimeError, "Unknown"):
            c.submit(["other"], tokens)
        self.assertFalse(list(self.directory.glob("rounds_*.jsonl")))

    def test_copy_is_enqueued_before_existing_completion_event(self):
        tree = ast.parse((ROOT / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_copy_draft_token_ids_to_cpu"
        )
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        ns = dict(torch=torch)
        fake = ast.ClassDef(name="Runner", bases=[], keywords=[], body=[method], decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fake], type_ignores=[])), "copy", "exec"), ns)
        c = self.controller("cut", trace=False)
        tokens = self.predict(c, [[1, 1, 1]])
        c.copied_req_ids = None
        runner = ns["Runner"]()
        runner.drafter = SimpleNamespace(dspark_confidence=c)
        runner._draft_token_ids = torch.tensor(tokens)
        runner.num_spec_tokens = 3
        runner.use_async_scheduling = False
        runner.input_batch = SimpleNamespace(req_ids=["r0"])
        runner.draft_token_ids_cpu = torch.empty(1, 3, dtype=torch.long)
        runner.draft_token_ids_copy_stream = SimpleNamespace(wait_stream=lambda stream: None)
        ordered = []
        runner.draft_token_ids_event = SimpleNamespace(record=lambda: ordered.append(c.copied_req_ids))
        with patch.object(
            torch, "npu", SimpleNamespace(current_stream=lambda: None, stream=lambda s: nullcontext()), create=True
        ):
            runner._copy_draft_token_ids_to_cpu(None)
        self.assertEqual(ordered, [("r0",)])
        self.assertEqual(c.submit(["r0"], runner.draft_token_ids_cpu.tolist()), [tokens[0][:2]])

    def test_real_proposer_hook_uses_sampled_rows_not_anchor_rows(self):
        tree = ast.parse((ROOT / "vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py").read_text())
        assignments = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "getattr"
            and any(isinstance(a, ast.Constant) and a.value == "dspark_confidence" for a in n.value.args)
        ]
        self.assertEqual(len(assignments), 1)
        parent = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and assignments[0] in n.body)
        start = parent.body.index(assignments[0])
        nodes = parent.body[start : start + 2]
        received = []
        confidence = SimpleNamespace(predict=lambda model, hidden, tokens, meta: received.append(hidden.clone()))
        this = SimpleNamespace(
            dspark_confidence=confidence, dynamic_spec=None, model=object(), num_speculative_tokens=3
        )
        ns = dict(
            self=this,
            sample_hidden_states=torch.arange(12).reshape(3, 4),
            last_hidden_states=torch.zeros(4, 4),
            draft_token_ids=torch.tensor([[1, 2, 3, 4]]),
            sampling_metadata=META,
            num_blk=1,
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "hook", "exec"), ns)
        self.assertTrue(torch.equal(received[0], ns["sample_hidden_states"]))
        this.dspark_confidence = None
        this.dynamic_spec = SimpleNamespace(update=lambda **kwargs: received.append(kwargs["last_hidden_states"]))
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "hook", "exec"), ns)
        self.assertEqual(received[-1].shape, (3, 4))

    def test_trace_filter_excludes_warmup_and_other_jobs(self):
        c = self.controller()
        for rid in ("cmpl-warmup-0", "cmpl-measured-0"):
            tokens = self.predict(c, [[1, 1, 1]], [rid])[0]
            c.submit([rid], [tokens])
            c.verified([rid], {rid: tokens}, [[77]])
        report = dict(complete=True, cells=[dict(benchmark="gsm8k", temperature=0, request_ids=["cmpl-measured"])])
        out = reporter.accuracy_report(self.directory, report)
        self.assertEqual(out["excluded_unmeasured_rounds"], 1)
        self.assertEqual(out["cells"][0]["full_block_accuracy"]["rounds"], 1)

    def test_accuracy_exact_under_over_and_capped_distinction(self):
        result = reporter.length_accuracy([(3, 3), (1, 3), (3, 1), (0, 0)])
        self.assertEqual((result["exact_pct"], result["under_pct"], result["over_pct"]), (50, 25, 25))
        self.assertEqual(result["mae"], 1)

    def test_config_rejects_invalid_modes_and_ranges(self):
        self.assertEqual(Config().mode, "off")
        for kwargs in (
            dict(mode="bad"),
            dict(budget_threshold=float("nan")),
            dict(budget_threshold=2),
            dict(min_verify_tokens=0),
            dict(trace_directory="/tmp/x"),
            dict(typo=True),
        ):
            with self.assertRaises(ValidationError):
                Config(**kwargs)

    def fixture_args(self):
        target, draft = self.directory / "target", self.directory / "draft"
        target.mkdir(exist_ok=True)
        draft.mkdir(exist_ok=True)
        (target / "config.json").write_text(json.dumps(dict(model_type="qwen3", hidden_size=16, vocab_size=32)))
        (draft / "config.json").write_text(
            json.dumps(
                dict(
                    speculators_model_type="dspark",
                    block_size=4,
                    sample_from_anchor=False,
                    draft_vocab_size=32,
                    enable_confidence_head=True,
                    transformer_layer_config=dict(hidden_size=16, vocab_size=32),
                )
            )
        )
        for path in (target, draft):
            (path / "model.safetensors").write_bytes(b"identity-test-only")
        return SimpleNamespace(
            spec_main=ROOT,
            target=target,
            draft=draft,
            npu="6",
            port=8217,
            memory_utilization=0.8,
            max_model_len=128,
            max_batched_tokens=128,
            mode="reference",
            target_execution="graph",
            served_model_name="test",
            graph_metrics=0,
            check_only=1,
            dspark_confidence_mode="off",
            dspark_confidence_initial_budget=2,
            comparison_profile="matched_reference",
            head_execution="reference",
            draft_enforce_eager="1",
        )

    def test_launcher_modes_preserve_checkpoint_and_execution_settings(self):
        args = self.fixture_args()
        records = []
        for mode in ("off", "observe", "cut"):
            args.dspark_confidence_mode = mode
            _, command, env, record = launcher.command_and_record(args)
            self.assertEqual(record["compilation_config"]["cudagraph_capture_sizes"], [2, 3, 4])
            self.assertEqual(env["VLLM_ASCEND_DSPARK_HEAD_GRAPH"], "off")
            self.assertEqual(record["additional_config"]["dspark_confidence_config"]["mode"], mode)
            self.assertIn("--no-async-scheduling", command)
            records.append(record)
        for key in ("speculative_config", "compilation_config", "target_sha256", "draft_sha256", "source_sha256"):
            self.assertEqual(records[0][key], records[2][key])
        self.assertNotEqual(records[0]["target_sha256"], records[0]["draft_sha256"])

    def benchmark(self, mode):
        args = self.fixture_args()
        args.dspark_confidence_mode = mode
        _, _, _, server = launcher.command_and_record(args)
        return dict(
            complete=True,
            manifest_sha256="prompts",
            eval_code_sha256="same-code",
            server_manifest=server,
            comparison_settings=dict(
                runtime_identity=dict(additional_config=copy.deepcopy(server["additional_config"]))
            ),
            cells=[
                dict(
                    benchmark="gsm8k",
                    temperature=0,
                    concurrency=1,
                    sampling={"temperature": 0},
                    prompt_hashes=["p"],
                    output_hashes=["o"],
                    summary=dict(
                        requests=1,
                        measurement_valid=True,
                        tokens_per_second=100 if mode == "off" else 120,
                        acceptance_length_with_bonus_pooled=5,
                    ),
                )
            ],
        )

    def test_speedup_requires_matching_outputs_settings_and_no_trace(self):
        baseline, cut = self.benchmark("off"), self.benchmark("cut")
        row = reporter.compare_reports(baseline, cut, "cut")[0]
        self.assertTrue(row["comparison_valid"], row["issues"])
        self.assertAlmostEqual(row["speedup"], 1.2)
        for change in ("output", "weights", "trace", "runtime"):
            altered = copy.deepcopy(cut)
            if change == "output":
                altered["cells"][0]["output_hashes"] = ["different"]
            elif change == "weights":
                altered["server_manifest"]["draft_sha256"] = "different"
            elif change == "trace":
                altered["server_manifest"]["dspark_confidence"]["trace_directory"] = "/tmp/debug"
            else:
                altered["comparison_settings"]["runtime_identity"]["additional_config"]["enable_reduce_sample"] = True
            row = reporter.compare_reports(baseline, altered, "cut")[0]
            self.assertFalse(row["comparison_valid"])
            self.assertIsNone(row["speedup"])


if __name__ == "__main__":
    unittest.main()
