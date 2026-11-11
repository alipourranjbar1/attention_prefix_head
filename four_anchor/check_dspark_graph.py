"""Run on a trainer NPU before launching the teacher; checks values and gradients."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .dspark_loss_graph import distribution_loss, graph_status


def eager(logits, confidence, p, labels, weights, mix, alpha):
    logq = logits.float().log_softmax(-1)
    overlap = torch.minimum(logq.exp(), p).sum(-1).clamp(0., 1.)
    ce = -logq.gather(1, labels[:, None]).squeeze(1)
    kl = F.kl_div(logq, p, reduction="none").sum(-1) if mix.get("kl_div", 0.) else torch.zeros_like(ce)
    bce = F.binary_cross_entropy_with_logits(confidence.float(), overlap.detach(), reduction="none")
    return ((mix.get("ce", 0.) * ce + mix.get("tv", 0.) * (1 - overlap)
             + mix.get("kl_div", 0.) * kl + alpha * bce) * weights).sum()


def check(device, rows, vocab, graph=True):
    torch.manual_seed(71)
    mix, alpha = {"ce": .1, "tv": .9}, 1.
    errors = []
    # Full and partial chunks, new contents at the same shape, two outstanding
    # forward nodes, and a non-unit upstream gradient all matter for replay.
    for size in (rows, max(1, rows - 3), rows):
        logits = torch.randn(size, vocab, device=device, dtype=torch.bfloat16).requires_grad_()
        confidence = torch.randn(size, device=device, dtype=torch.bfloat16).requires_grad_()
        p = torch.randn(size, vocab, device=device).softmax(-1)
        labels = p.argmax(-1)
        weights = torch.rand(size, device=device)
        weights[::5] = 0
        x, c = logits.detach().clone().requires_grad_(), confidence.detach().clone().requires_grad_()
        actual = distribution_loss(logits, confidence, p, labels, weights, mix, alpha, rows, graph)[0]
        # Second call must not mutate the first call's saved gradients.
        other = distribution_loss(logits * .5, confidence * .25, p, labels, weights, mix, alpha, rows, graph)[0]
        expected = eager(x, c, p, labels, weights, mix, alpha)
        expected2 = eager(x * .5, c * .25, p, labels, weights, mix, alpha)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-5)
        ((actual + other) * .37).backward()
        ((expected + expected2) * .37).backward()
        torch.testing.assert_close(logits.grad, x.grad, rtol=.02, atol=2e-5)
        torch.testing.assert_close(confidence.grad, c.grad, rtol=.02, atol=2e-3)
        errors.append(float((logits.grad - x.grad).abs().max()))

    def sync():
        if str(device).startswith("npu"):
            torch.npu.synchronize()

    def measure(use_graph):
        for _ in range(3):
            logits.grad = confidence.grad = None
            value = (distribution_loss(logits, confidence, p, labels, weights, mix, alpha, rows, True)[0]
                     if use_graph else eager(logits, confidence, p, labels, weights, mix, alpha))
            value.backward()
        sync()
        start = time.perf_counter()
        for _ in range(10):
            logits.grad = confidence.grad = None
            value = (distribution_loss(logits, confidence, p, labels, weights, mix, alpha, rows, True)[0]
                     if use_graph else eager(logits, confidence, p, labels, weights, mix, alpha))
            value.backward()
        sync()
        return (time.perf_counter() - start) / 10

    result = {"status": "passed", "device": str(device), "rows": rows, "vocab": vocab,
              "max_logit_gradient_error": max(errors), "scope": "loss only; not end-to-end training"}
    if graph:
        result.update(eager_seconds=measure(False), graph_seconds=measure(True), **graph_status())
        result["loss_only_speedup"] = result["eager_seconds"] / result["graph_seconds"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch_npu
    torch.npu.set_device(0)
    result = check("npu:0", args.rows, args.vocab)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["loss_only_speedup"] < 1:
        print("Loss graph is slower in this check. LOSS_GRAPH=off is available; compare full training steps.", flush=True)


if __name__ == "__main__":
    main()
