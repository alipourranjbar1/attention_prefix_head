"""One-pass suite, kernel preflight, shell dry-run and real chunked HTTP checks."""

import contextlib
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Evaluator"))
sys.path.insert(0, str(Path(__file__).parent))
import dspark_once_control as control
import dspark_benchmark_suite as suite
import evaluator
import test_dspark_full_suite as fixtures


class OnePassTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def fixture_report(self, changed):
        helper = fixtures.FullSuiteTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.prepare_report_fixture(128)
        metadata = suite.read(helper.path / "suite.json")
        metadata["settings"]["repeats"] = 1
        evaluator.write_json(helper.path / "suite.json", metadata)
        for mode in ("cut", "off"):
            path = helper.write_run(mode, 0, changed=changed and mode == "cut")
            evaluator.write_json(helper.path / f"runs_{mode}.json", [str(path.relative_to(helper.path))])
        report = suite.build_report(helper.path)
        return helper.path, report

    def test_single_pass_has_no_invented_variance_and_preserves_mismatches(self):
        for changed in (False, True):
            with self.subTest(changed=changed):
                path, report = self.fixture_report(changed)
                self.assertTrue(report["complete"])
                for row in report["benchmarks"]:
                    self.assertEqual(row["off_repeats"], 1)
                    self.assertIsNone(row["off_wall_tps_stdev"])
                    self.assertIsNone(row["cut_wall_tps_stdev"])
                    self.assertEqual(row["strict_comparison_valid"], not (changed and row["benchmark"] == "gsm8k"))
                with contextlib.redirect_stdout(io.StringIO()):
                    status = control.finish(path)
                self.assertEqual(status, 2 if changed else 0)
                self.assertTrue((path / "ON/run_info.json").is_file())
                self.assertTrue((path / "OFF/run_info.json").is_file())
                self.assertEqual(suite.read(path / "status.json")["strict_output_comparison_valid"], not changed)

    def test_preflight_does_not_accept_triton_without_attention_operators(self):
        root = self.path
        modules = {
            "torch": SimpleNamespace(__version__="fixture"), "torch_npu": SimpleNamespace(__version__="fixture"),
            "vllm": SimpleNamespace(__file__=str(root / "vllm/vllm/__init__.py")),
            "vllm_ascend": SimpleNamespace(__file__=str(root / "vllm-ascend/vllm_ascend/__init__.py"),
                                           batch_invariant=SimpleNamespace(HAS_ASCENDC_BATCH_INVARIANT=False, HAS_TRITON=True)),
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, control.EXPECTED_ENV):
            with self.assertRaisesRegex(ValueError, "attention replacement"):
                control.preflight(root, root / "preflight.json")
        self.assertFalse((root / "preflight.json").exists())

    def test_actual_shell_check_only_has_one_on_then_one_off_at_128(self):
        target, draft, conda = self.path / "target", self.path / "draft", self.path / "conda"
        for path in (target, draft, conda / "bin"):
            path.mkdir(parents=True)
        (conda / "bin/python").symlink_to(sys.executable)
        (target / "config.json").write_text(json.dumps(dict(model_type="qwen3", hidden_size=16, vocab_size=32, max_position_embeddings=8192)))
        (target / "model.safetensors").write_bytes(b"fixture metadata only")
        (draft / "config.json").write_text(json.dumps(dict(model_type="qwen3", architectures=["Qwen3DSparkModel"],
                                                         block_size=7, hidden_size=16, vocab_size=32, enable_confidence_head=True)))
        (draft / "model.safetensors").write_bytes(b"fixture metadata only")
        manifest = dict(format=3, thinking=False, target_config_sha256=evaluator.file_hash(target / "config.json"), datasets={})
        for name in suite.BENCHMARKS:
            manifest["datasets"][name] = dict(source_rows=1, source={"fixture": True}, results=[
                dict(source_row=0, prompt_token_ids=[1, 2], sha256=evaluator.digest([1, 2]))])
        evaluator.write_json(self.path / "prompts.json", manifest)
        results = self.path / "results"
        env = dict(os.environ, CHECK_ONLY="1", ROOT=str(ROOT), RESULTS_ROOT=str(results), CONDA_PREFIX=str(conda),
                   TARGET_MODEL=str(target), DRAFT_MODEL=str(draft), PROMPTS_SOURCE=str(self.path / "prompts.json"))
        result = subprocess.run(["bash", str(ROOT / "Evaluator/run_dspark_on_then_off_once.sh")], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = Path((results / "latest_on_then_off.txt").read_text().strip())
        metadata = suite.read(output / "suite.json")
        self.assertEqual(metadata["settings"]["repeats"], 1)
        self.assertEqual(metadata["settings"]["concurrency"], 128)
        self.assertEqual((output / "prompts.json").read_bytes(), (self.path / "prompts.json").read_bytes())
        self.assertFalse(list(output.glob("runs_*.json")))
        self.assertLess(result.stdout.index("configuration: cut"), result.stdout.index("configuration: off"))
        for mode in ("cut", "off"):
            paths = list(output.glob(f"server_{mode}_*/launch.json"))
            self.assertEqual(len(paths), 1)
            launch = suite.read(paths[0])
            self.assertEqual(launch["max_num_seqs"], 128)
            self.assertEqual(launch["proposals"], 7)
            self.assertEqual(launch["determinism_environment"], control.EXPECTED_ENV)
            self.assertEqual(launch["speculative_config"]["rejection_sample_method"], "standard")
            # Current AscendConfig forbids this obsolete top-level key.
            self.assertNotIn("enable_batch_invariant", launch["additional_config"])
            self.assertEqual(launch["dspark_confidence"]["mode"], mode)


@unittest.skipUnless(importlib.util.find_spec("requests"), "requests is needed for the real HTTP streaming test")
class StreamingTests(unittest.TestCase):
    def test_short_sse_event_is_delivered_without_waiting_for_1024_bytes(self):
        first_read = threading.Event()
        result = {}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_GET(self):
                payload = json.dumps({"data": [{"id": "fixture"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def chunk(payload):
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()

                first = b'data: {"id":"fixture","choices":[{"index":0,"token_ids":[3],"text":"a"}]}\n\n'
                chunk(first[:20])
                chunk(first[20:])
                result["early_delivery"] = first_read.wait(3)
                chunk(b'data: {"id":"fixture","choices":[{"index":0,"token_ids":[4],"text":"b","finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n')
                chunk(b'data: [DONE]\n\n')
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = evaluator.Client(f"http://127.0.0.1:{server.server_port}", timeout=5)
            session = client._request_session()
            real_post = session.post

            def post(*args, **kwargs):
                response = real_post(*args, **kwargs)
                real_lines = response.iter_lines

                def lines(*args, **kwargs):
                    result["chunk_size"] = kwargs["chunk_size"]
                    for line in real_lines(*args, **kwargs):
                        first_read.set()
                        yield line
                response.iter_lines = lines
                return response

            with patch.object(session, "post", side_effect=post):
                row = client.complete([1, 2, 3], {"temperature": 0})
            self.assertTrue(result["early_delivery"], "First event was buffered until later output")
            self.assertEqual(result["chunk_size"], 1024)
            self.assertEqual(row["output_token_ids"], [3, 4])
            self.assertAlmostEqual(row["latency_seconds"], row["request_finished_monotonic_s"] - row["request_started_monotonic_s"])
            self.assertLessEqual(row["request_started_monotonic_s"], row["first_token_monotonic_s"])
            self.assertLessEqual(row["last_token_monotonic_s"], row["request_finished_monotonic_s"])
            client.session.close()
            session.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(5)


if __name__ == "__main__":
    unittest.main()
