"""CPU tests: objective, boundaries, trace pairing and frozen-head training."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[2] / "src/speculators/models/dflash_prefix/acceptance_head.py"
spec = importlib.util.spec_from_file_location("acceptance_test_module", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class AcceptanceTests(unittest.TestCase):
    def test_exact_is_best_and_equal_distance_under_is_more_costly(self):
        target = torch.tensor([6])
        losses = []
        for predicted in (6, 5, 7, 1):
            logits = torch.zeros(1, 16)
            logits[0, predicted] = 4
            logits.requires_grad_()
            loss = m.boundary_loss(logits, target)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            losses.append(float(loss.detach()))
        self.assertLess(losses[0], losses[2])
        self.assertGreater(losses[1], losses[2])
        self.assertGreater(losses[3], losses[1])

    def test_features_follow_selected_candidate_not_unary_argmax(self):
        hidden = torch.zeros(1, 2, 3)
        ids = torch.tensor([[[10, 11], [12, 13]]])
        unary = torch.tensor([[[5., 3.], [9., 2.]]])
        scores = torch.tensor([[[4., 6.], [10., 4.]]])
        selected = torch.tensor([[11, 12]])
        f = m.build_features(hidden, ids, unary, selected, scores)
        self.assertEqual(tuple(f.shape), (1, 2, 8))
        self.assertEqual(f[0, 0, 3].item(), -2)
        self.assertEqual(f[0, 0, 6].item(), 3)
        self.assertEqual(f[0, 1, 7].item(), 1)

    def test_rounding_never_exceeds_full_block_or_returns_zero(self):
        self.assertEqual(m.round_lengths(torch.tensor([0, 1, 2, 7, 15]),
                         torch.tensor([1, 4, 8, 15])).tolist(), [1, 1, 4, 8, 15])

    def test_full_block_labels_first_failure_all_accept_and_censoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("draft", "target"):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({"eos_token_id": 99}))
                (folder / "model.safetensors").write_bytes(b"identity-test-only")
            runtime = m.AcceptanceRuntime(dict(mode="collect", trace_dir=str(root / "traces")),
                root / "draft", root / "target", 3, "cpu")
            tokens = [10, 11, 12]
            def round(output, scheduled=None):
                runtime.predict(torch.zeros(1, 3, 9), torch.tensor([tokens]))
                runtime.submit(["request"], [tokens], [1, 2, 3])
                runtime.observe(["request"], [output], {"request": tokens if scheduled is None else scheduled})
            round([20])
            round([10, 11, 12, 20])
            round([10, 20], [10])  # truncated proposal cannot label a full block
            round([10, 99])        # EOS censoring
            rows = [torch.load(p, weights_only=True) for p in sorted(runtime.directory.glob("*.pt"))]
            self.assertEqual([x["accepted_length"] for x in rows], [0, 3])
            self.assertEqual(rows[0]["prompt_key"], rows[1]["prompt_key"])

            head = m.AcceptedLengthHead(4, 3)
            for parameter in head.parameters():
                parameter.data.zero_()
            head.classifier[-1].bias.data[1] = 10
            head_path = root / "head.pt"
            torch.save(dict(identity=runtime.identity, architecture=head.architecture,
                            state_dict=head.state_dict()), head_path)
            adaptive = m.AcceptanceRuntime(dict(mode="adaptive", head_path=str(head_path),
                verify_lengths=[1, 3]), root / "draft", root / "target", 3, "cpu")
            adaptive.predict(torch.zeros(1, 3, 9), torch.tensor([tokens]))
            self.assertEqual(adaptive.submit(["request"], [tokens], [1, 2, 3]), [[10]])
            self.assertTrue(all(not p.requires_grad for p in adaptive.head.parameters()))
            changed = torch.load(head_path, weights_only=True)
            changed["identity"]["draft_sha256"] = "wrong"
            torch.save(changed, head_path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                m.AcceptanceRuntime(dict(mode="adaptive", head_path=str(head_path)),
                    root / "draft", root / "target", 3, "cpu")

    def test_mlp_learns_small_separable_boundary_problem(self):
        torch.manual_seed(7)
        head = m.AcceptedLengthHead(4, 3, projection=8, width=16)
        x = torch.randn(16, 3, 9)
        labels = torch.tensor([0] * 8 + [3] * 8)
        x[:8, :, -1] = -3
        x[8:, :, -1] = 3
        opt = torch.optim.AdamW(head.parameters(), lr=.02)
        first = float(m.boundary_loss(head(x), labels).detach())
        for _ in range(60):
            loss = m.boundary_loss(head(x), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(m.boundary_loss(head(x), labels).detach()), first / 5)
        self.assertTrue(torch.equal(head(x).argmax(-1), labels))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
