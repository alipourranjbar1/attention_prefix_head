"""Check the candidate loss on the selected device before full model allocation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from .dspark_model import FourAnchorDSpark, DSparkDraftConfig
from .dspark_loss import projected_dspark_loss
from .runtime import resolve_device


def check(device):
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=1024, hidden_size=512, intermediate_size=1024,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, max_position_embeddings=128)
    target = Qwen3ForCausalLM(cfg).to(device).eval().requires_grad_(False)
    draft = FourAnchorDSpark(DSparkDraftConfig("synthetic", [0, 1], 1023,
        num_draft_layers=2, num_proposals=15, anchor_layout="bonus_addon", format_version=4,
        markov_rank=32), target)
    records = []
    for low_params in (False, True):
        reference = deepcopy(draft).to(dtype=torch.bfloat16 if low_params else torch.float32)
        candidate = deepcopy(reference)
        candidate.head_loss_backend = "cached_vjp"
        h = torch.randn(2, 15, 512, device=device, dtype=torch.bfloat16, requires_grad=True)
        hr = h.detach().clone().requires_grad_(True)
        prev = torch.randint(0, 1024, (2, 15), device=device)
        p = torch.randn(40, 1024, device=device).softmax(-1)
        indices = torch.randint(0, 40, (2, 15), device=device)
        weights = torch.rand(2, 15, device=device)
        weights[:, -3:] = 0
        args = prev, (p, p.argmax(-1)), indices, weights, {"ce": .1, "tv": .9}, 1., 7
        with torch.autocast(device.type, dtype=torch.bfloat16):
            actual = projected_dspark_loss(candidate, h, *args)
            expected = projected_dspark_loss(reference, hr, *args)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-5)
        actual[0].backward()
        expected[0].backward()
        comparisons = [("hidden", h.grad, hr.grad)]
        comparisons += [(n, a.grad, b.grad) for (n, a), (_, b) in
                        zip(candidate.named_parameters(), reference.named_parameters()) if a.grad is not None]
        errors = {}
        for name, a, b in comparisons:
            if b is None or not torch.isfinite(a).all():
                raise AssertionError(f"Invalid cached gradient: {name}")
            af, bf = a.float(), b.float()
            ratio = float((af - bf).norm() / bf.norm().clamp_min(1e-10))
            if ratio > .015:
                raise AssertionError(f"Cached gradient relative L2 error {name}: {ratio:.5f} > 0.015")
            errors[name] = ratio
        records.append({"parameter_dtype": str(next(candidate.parameters()).dtype),
                        "gradient_relative_l2_errors": errors})
    return {"status": "passed", "device": str(device), "checks": records,
            "note": "BF16 tolerance test, not bitwise equality or a throughput benchmark"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = check(resolve_device(args.device))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
