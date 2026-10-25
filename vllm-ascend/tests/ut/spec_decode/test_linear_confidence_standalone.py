"""CPU tests runnable directly without importing the NPU plugin's conftest."""
import ast
from contextlib import nullcontext
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "speculators/src/speculators/models/dflash_prefix"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the actual lightweight modules without the training package dependencies.
pkg = ModuleType("confidence_test_package")
pkg.__path__ = [str(PACKAGE)]
sys.modules[pkg.__name__] = pkg
runtime = load(pkg.__name__ + ".linear_confidence", PACKAGE / "linear_confidence.py")
launcher = load("confidence_launcher", ROOT / "speculators/scripts/dflash_prefix_compiled/target_graph_server.py")


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "final.pt"
        head = runtime.AcceptedLengthHead(4, 3, 2, 2)
        for param in head.parameters():
            param.data.zero_()
        self.saved = dict(architecture=head.architecture, state_dict=head.state_dict(),
                          identity=dict(schema=runtime.FEATURE_SCHEMA,
                                        decoding="greedy_default_penalties_v1"))
        self.tokens = torch.tensor([[11, 22, 33]])
        self.features = torch.zeros(1, 3, 9, dtype=torch.bfloat16)

    def predictor(self, count=2):
        self.saved["state_dict"]["boundary_classifier.2.bias"].zero_()
        self.saved["state_dict"]["boundary_classifier.2.bias"][count] = 5
        torch.save(self.saved, self.path)
        return runtime.LinearConfidence(self.path, "cpu", 4, 3, max_batch_size=128)

    def test_every_boundary_and_one_time_consumption(self):
        for count in range(4):
            predictor = self.predictor(count)
            predictor.predict(self.features, self.tokens)
            predictor.queue_copy(["r"])
            self.assertEqual(predictor.truncate(["r"], [[11, 22, 33]]),
                             [[11, 22, 33][:count]])
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                predictor.truncate(["r"], [[11, 22, 33]])

    def test_mismatched_tokens_fail_closed(self):
        predictor = self.predictor()
        predictor.predict(self.features, self.tokens)
        predictor.queue_copy(["r"])
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            predictor.truncate(["r"], [[11, 22, 99]])
        self.assertIsNone(predictor.pending)

    def test_nonfinite_logits(self):
        predictor = self.predictor()
        features = self.features.clone()
        features[0, 0, 0] = float("nan")
        predictor.predict(features, self.tokens)
        predictor.queue_copy(["r"])
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            predictor.truncate(["r"], [[11, 22, 33]])

    def test_no_proposal_and_reset(self):
        predictor = self.predictor()
        predictor.predict(self.features, self.tokens)
        predictor.queue_copy(["r"])
        self.assertEqual(predictor.truncate(["r"], [[]]), [[]])
        predictor.predict(self.features, self.tokens)
        predictor.queue_copy(["r"])
        predictor.reset()
        self.assertIsNone(predictor.pending)

    def test_checkpoint_validation(self):
        self.predictor()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            runtime.LinearConfidence(self.path, "cpu", 5, 3)
        self.saved["identity"]["schema"] = "wrong"
        with self.assertRaisesRegex(ValueError, "schema"):
            self.predictor()

    def test_runner_boundary_and_disabled_path(self):
        # Execute the real override with only its base runner and record mocked.
        source = ast.parse((ROOT / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "take_draft_token_ids")
        method.returns = None
        fake = ast.ClassDef(name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())],
                            keywords=[], body=[method], decorator_list=[])
        out = SimpleNamespace(req_ids=["r"], draft_token_ids=[[11, 22, 33]])
        class Base:
            def take_draft_token_ids(self):
                return out
        namespace = dict(Base=Base, DraftTokenIds=SimpleNamespace, logger=logging.getLogger(__name__))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fake], type_ignores=[])), "runner", "exec"), namespace)
        runner = namespace["Runner"]()
        runner.drafter = SimpleNamespace()
        self.assertIs(runner.take_draft_token_ids(), out)
        predictor = self.predictor(1)
        predictor.predict(self.features, self.tokens)
        predictor.queue_copy(["r"])
        runner.drafter.linear_confidence = predictor
        self.assertEqual(runner.take_draft_token_ids().draft_token_ids, [[11]])

    def test_proposer_runs_head_after_selector_and_skips_dummy(self):
        source = ast.parse((ROOT / "vllm-ascend/vllm_ascend/spec_decode/dflash_prefix_proposer.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_draft_token_ids")
        fake = ast.ClassDef(name="Proposer", bases=[], keywords=[], body=[method], decorator_list=[])
        ns = dict(torch=torch, logger=logging.getLogger(__name__))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fake], type_ignores=[])), "proposer", "exec"), ns)
        proposer = ns["Proposer"]()
        proposer.linear_confidence = self.predictor(2)
        proposer.acceptance_collector = None
        proposer._prefix_in_dummy_run = False
        proposer._prefix_disable_selector = False
        proposer._compiled_prefix = True
        proposer._prefix_inference_mode = "reference"
        proposer.num_speculative_tokens = 3
        proposer.max_batch_size = 1
        proposer.prefix_top_k = 2
        proposer.input_ids = torch.tensor([7])
        proposer._prefix_anchor_indices = torch.tensor([0])
        candidates = torch.tensor([[[11, 12], [22, 23], [33, 34]]])
        scores = torch.tensor([[[2., 1.], [2., 1.], [2., 1.]]])
        calls = []
        def select(*args, **kwargs):
            calls.append(kwargs)
            return (self.tokens, scores) if kwargs.get("return_scores") else self.tokens
        proposer._prefix_inference_runtime = SimpleNamespace(select=select)
        head = SimpleNamespace(prepare_tables=lambda *a, **k: None,
                               greedy_walk=lambda tables: self.tokens)
        proposer.model = SimpleNamespace(
            compute_candidates=lambda hidden: (candidates, scores),
            model=SimpleNamespace(prefix_head=head, embed_tokens=SimpleNamespace(weight=None)))
        metadata = SimpleNamespace(all_greedy=True, no_penalties=True,
                                   allowed_token_ids_mask=None, bad_words_token_ids={})
        module_name = "speculators.models.dflash_prefix.acceptance_features"
        features_module = sys.modules[pkg.__name__ + ".acceptance_features"]
        with patch.dict(sys.modules, {module_name: features_module}):
            selected, _ = proposer.compute_draft_token_ids(torch.zeros(3, 4), metadata)
            self.assertTrue(calls[-1]["return_scores"])
            self.assertEqual(selected.tolist(), [11, 22, 33])
            proposer.linear_confidence.queue_copy(["r"])
            self.assertEqual(proposer.linear_confidence.truncate(["r"], [[11,22,33]]), [[11,22]])
            proposer._prefix_in_dummy_run = True
            proposer._prefix_dummy_logged = True
            proposer.compute_draft_token_ids(torch.zeros(3, 4))
            self.assertIsNone(proposer.linear_confidence.pending)
            self.assertEqual(len(calls), 1)
            proposer._prefix_in_dummy_run = False
            proposer.linear_confidence = None
            proposer.compute_draft_token_ids(torch.zeros(3, 4), metadata)
            self.assertNotIn("return_scores", calls[-1])

    def test_batch_lengths_request_reordering_and_reuse(self):
        predictor = self.predictor()
        class FixedHead(torch.nn.Module):
            num_proposals = 3
            def forward(self, features):
                return torch.tensor([[9., 0., 0., 0.], [0., 9., 0., 0.],
                                     [0., 0., 0., 9.]])
        predictor.head = FixedHead()
        tokens = torch.tensor([[10,11,12], [20,21,22], [30,31,32]])
        predictor.predict(self.features.expand(3,-1,-1), tokens)
        predictor.queue_copy(["a", "b", "c"])
        self.assertEqual(predictor.truncate(["c", "a", "b"],
                         [[30,31,32], [10,11,12], [20,21,22]]), [[30,31,32], [], [20]])
        # Next round can have a completely different request set and batch size.
        predictor = self.predictor(2)
        predictor.predict(self.features, self.tokens)
        predictor.queue_copy(["new"])
        self.assertEqual(predictor.truncate(["new"], [[11,22,33]]), [[11,22]])

    def test_missing_copy_unknown_request_and_duplicate(self):
        predictor = self.predictor()
        predictor.predict(self.features, self.tokens)
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            predictor.truncate(["r"], [[11,22,33]])
        predictor.predict(self.features, self.tokens)
        predictor.queue_copy(["r"])
        with self.assertRaisesRegex(RuntimeError, "Unknown"):
            predictor.truncate(["other"], [[11,22,33]])
        predictor.predict(self.features.expand(2,-1,-1), self.tokens.expand(2,-1))
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            predictor.queue_copy(["r", "r"])

    def test_shared_budget_assigns_more_to_high_survival(self):
        predictor = self.predictor()
        predictor.policy = "budget"
        predictor.budget_per_request = 2
        # Request a confidently accepts all three; b confidently accepts zero.
        logits = torch.tensor([[0.,0.,0.,20.], [20.,0.,0.,0.]])
        self.assertEqual(predictor.select_lengths(logits).tolist(), [3,1])
        for batch in (1,2,16,128):
            for budget in (1,2,3):
                predictor.budget_per_request = budget
                lengths = predictor.select_lengths(torch.randn(batch,4))
                self.assertEqual(int(lengths.sum()), batch*budget)
                self.assertTrue(bool(((lengths >= 1) & (lengths <= 3)).all()))

    def test_shared_copy_precedes_existing_event(self):
        source = ast.parse((ROOT / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_copy_draft_token_ids_to_cpu")
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        fake = ast.ClassDef(name="Runner", bases=[], keywords=[], body=[method], decorator_list=[])
        ns = dict(torch=torch)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fake], type_ignores=[])), "copy", "exec"), ns)
        runner = ns["Runner"]()
        predictor = self.predictor(2)
        predictor.predict(self.features.expand(2,-1,-1), self.tokens.expand(2,-1))
        runner.drafter = SimpleNamespace(linear_confidence=predictor)
        runner._draft_token_ids = self.tokens.expand(2,-1)
        runner.num_spec_tokens = 3
        runner.use_async_scheduling = False
        runner.input_batch = SimpleNamespace(req_ids=["a","b"])
        runner.draft_token_ids_cpu = torch.empty(2,3,dtype=torch.long)
        runner.draft_token_ids_copy_stream = SimpleNamespace(wait_stream=lambda stream: None)
        recorded = []
        runner.draft_token_ids_event = SimpleNamespace(record=lambda: recorded.append(predictor.copy_queued))
        npu = SimpleNamespace(current_stream=lambda: None, stream=lambda stream: nullcontext())
        with patch.object(torch, "npu", npu, create=True):
            runner._copy_draft_token_ids_to_cpu(None)
        self.assertEqual(recorded, [True])
        self.assertEqual(predictor.truncate(["a","b"], runner.draft_token_ids_cpu.tolist()),
                         [[11,22],[11,22]])

    def test_local_v2_scores_reference_and_fast(self):
        v2 = load(pkg.__name__ + ".inference_v2", PACKAGE / "inference_v2.py")
        torch.manual_seed(42)
        head = v2.LocalPrefixSelector(20, 4, rank=8, top_k=2,
                                     max_proposals=3, heads=2, layers=1).eval()
        hidden = torch.randn(3, 3, 4)
        ids = torch.tensor([[[1, 2], [3, 4], [5, 6]]]).expand(3,-1,-1)
        unary = torch.randn(3, 3, 2)
        anchors = torch.tensor([0,1,2])
        embeddings = torch.randn(20, 4)
        with torch.no_grad():
            tables = head.prepare_tables(hidden, ids, unary, anchors, embedding_weight=embeddings)
            expected, scores = head.walk(tables, return_scores=True)
            for mode in ("reference", "validate", "fast"):
                env = dict(DFLASH_PREFIX_INFERENCE=mode, DFLASH_PREFIX_NPU_GRAPH="0",
                           DFLASH_PREFIX_GRAPH_SCOPE="walk", DFLASH_PREFIX_FUSION="off",
                           DFLASH_PREFIX_TOKEN_CACHE="0", DFLASH_PREFIX_VALIDATE_STEPS="1")
                with patch.dict(v2.os.environ, env):
                    selector = v2.PrefixInferenceRuntime(head, logging.getLogger(__name__))
                # Both initial parity-validation and steady-state paths.
                for _ in range(2):
                    actual, actual_scores = selector.select(hidden, ids, unary, anchors,
                                                           embedding_weight=embeddings, return_scores=True)
                    self.assertTrue(torch.equal(actual, expected))
                    torch.testing.assert_close(actual_scores, scores, rtol=.005, atol=.02)
                    plain = selector.select(hidden, ids, unary, anchors, embedding_weight=embeddings)
                    self.assertTrue(torch.equal(plain, expected))

    def test_launcher_enables_all_lengths_and_clears_stale_flag(self):
        self.predictor()
        draft, target = Path(self.tmp.name)/"draft", Path(self.tmp.name)/"target"
        for folder in (draft, target):
            folder.mkdir()
            (folder/"config.json").write_text("{}")
        args = SimpleNamespace(spec_main=ROOT, draft=draft, target=target,
            linear_confidence=True, linear_confidence_head=self.path,
            draft_enforce_eager="1", npu="0", port=8116, memory_utilization=.8,
            max_model_len=128, max_batched_tokens=128, mode="reference",
            target_execution="graph", served_model_name="test", graph_metrics=0, check_only=1)
        spec = dict(num_speculative_tokens=3, enforce_eager=True, method="dflash")
        with patch.object(launcher, "checkpoint", return_value=(draft, target, "local_prefix_v2", 4, spec)):
            _, _, env, record = launcher.command_and_record(args)
            self.assertEqual(record["compilation_config"]["cudagraph_capture_sizes"], [1,2,3,4])
            self.assertEqual(env["VLLM_ASCEND_PREFIX_LINEAR_CONFIDENCE_HEAD"], str(self.path))
            args.linear_confidence = False
            args.linear_confidence_head = None
            with patch.dict(launcher.os.environ, {"VLLM_ASCEND_PREFIX_LINEAR_CONFIDENCE_HEAD": "stale"}):
                _, _, env, record = launcher.command_and_record(args)
            self.assertNotIn("VLLM_ASCEND_PREFIX_LINEAR_CONFIDENCE_HEAD", env)
            self.assertEqual(record["compilation_config"]["cudagraph_capture_sizes"], [4])
            args.max_num_seqs = 128
            args.target_execution = "eager"
            args.linear_confidence = True
            args.linear_confidence_head = self.path
            args.linear_confidence_policy = "budget"
            args.linear_confidence_budget = 2
            _, command, env, record = launcher.command_and_record(args)
            self.assertEqual(command[command.index("--max-num-seqs")+1], "128")
            self.assertEqual(env["VLLM_ASCEND_PREFIX_LINEAR_CONFIDENCE_POLICY"], "budget")
            self.assertEqual(record["max_num_seqs"], 128)
            args.target_execution = "graph"
            with self.assertRaisesRegex(ValueError, "eager"):
                launcher.command_and_record(args)



if __name__ == "__main__":
    unittest.main()
