from dataclasses import asdict
from pathlib import Path
import json

import torch

from .data import load_records, prompt_tokens
from .provenance import record, sha256
from .runtime import Decoder


def evaluate(runner, drafter, tokenizer, data, output, limit=100, max_new_tokens=256,
             warmup=2, checkpoint=None):
    """Paired evaluation on the SAME prompts, tokens, target, and output budget.

    one_pass is an ablation of this four-anchor checkpoint, not a vanilla DFlash
    benchmark. Both acceptance-without-correction and committed tokens are saved.
    """
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    settings = {"data": data, "limit": limit, "max_new_tokens": max_new_tokens,
                "warmup": warmup, "checkpoint": checkpoint,
                "checkpoint_sha256": sha256(Path(checkpoint) / "model.safetensors") if checkpoint else None}
    record(path, "eval", settings)
    records = load_records(data)
    decoder = Decoder(runner, drafter)
    device = next(runner.target.parameters()).device
    modes = ["ar", "one_pass", "two_pass"]
    rows = []
    count = min(limit, len(records))
    if count < 1:
        raise ValueError("Evaluation requires at least one prompt")
    first_ids = torch.tensor([prompt_tokens(records[0], tokenizer)], device=device)
    for _ in range(warmup):
        for mode in modes:
            decoder.generate(first_ids, min(16, max_new_tokens), mode)
    with (path / "per_prompt.jsonl").open("w") as stream:
        for i in range(count):
            ids = torch.tensor([prompt_tokens(records[i], tokenizer)], device=device)
            generated = {}
            # Rotate execution order to reduce systematic warm/cold-order bias.
            for mode in modes[i % 3:] + modes[:i % 3]:
                generated[mode] = decoder.generate(ids, max_new_tokens, mode)
            for mode in modes:
                result = generated[mode]
                equal = result.token_ids == generated["ar"].token_ids
                row = {"prompt_index": i, "mode": mode, **result.metrics(),
                       "matches_greedy_target": equal, "token_ids": result.token_ids,
                       "cycle_trace": [asdict(c) for c in result.cycles]}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
            print(json.dumps({"prompt": i, "matches": all(generated[m].token_ids == generated["ar"].token_ids for m in modes),
                              "two_pass": generated["two_pass"].metrics()}), flush=True)
    summary = {}
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        tokens = sum(r["output_tokens"] for r in subset)
        seconds = sum(r["seconds_including_prefill"] for r in subset)
        cycles = sum(r["cycles"] for r in subset)
        trace = [c for r in subset for c in r["cycle_trace"]]
        summary[mode] = {"prompts": len(subset), "tokens": tokens, "seconds": seconds,
                         "tokens_per_second": tokens / max(seconds, 1e-9),
                         "accepted_draft_per_cycle": sum(c["accepted_draft"] for c in trace) / max(cycles, 1),
                         "committed_per_cycle": sum(c["committed"] for c in trace) / max(cycles, 1),
                         "all_match_greedy_target": all(r["matches_greedy_target"] for r in subset)}
    for mode in modes:
        summary[mode]["speedup_vs_ar"] = summary[mode]["tokens_per_second"] / max(summary["ar"]["tokens_per_second"], 1e-9)
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not all(s["all_match_greedy_target"] for s in summary.values()):
        raise RuntimeError("Greedy parity failed; see per_prompt.jsonl before trusting speed or acceptance metrics")
    return summary
