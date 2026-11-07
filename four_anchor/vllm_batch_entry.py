"""Evaluate a queued workload with up to 32 active sequences in one engine."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .batch_inference import VERSION
from .vllm_entry import engine_options as single_options, server_command, worker_counters


def engine_options(args):
    if not 1 <= args.concurrency <= 32:
        raise ValueError("concurrency must be 1..32")
    if args.draft_graph or args.profile_steps or args.anchor_policy != "bonus_addon":
        raise ValueError("Require bonus_addon, DRAFT_GRAPH=0, PROFILE_STEPS=0")
    options = single_options(args)
    options.update(max_num_seqs=args.concurrency,
                   max_num_batched_tokens=max(args.max_model_len, args.concurrency * 16))
    if args.mode != "ar":
        options["worker_cls"] = "four_anchor.vllm_batch_ascend.BatchedFourAnchorWorker"
        options["speculative_config"]["model"] = "four_anchor.vllm_batch_ascend.BatchedFourAnchorProposer"
    return options


def summarize(rows, wall_seconds, concurrency):
    if wall_seconds <= 0:
        raise ValueError("Batch wall time must be positive")
    tokens = sum(r["output_tokens"] for r in rows)
    cycles = sum((r["speculative_decoding"] or {}).get("num_spec_steps", 0) for r in rows)
    accepted = sum((r["speculative_decoding"] or {}).get("num_accepted_draft_tokens", 0) for r in rows)
    return {"prompts": len(rows), "output_tokens": tokens, "seconds": wall_seconds,
        "tokens_per_second_including_prefill": tokens / wall_seconds,
        "concurrency": concurrency, "throughput_scope": "aggregate_output_tokens / generate_wall_seconds",
        "verification_cycles": cycles, "accepted_draft_per_verification": accepted / cycles if cycles else None,
        "acceptance_length_including_bonus": 1 + accepted / cycles if cycles else None,
        "acceptance_note": "Pooled request verification cycles before EOS/budget truncation; anchors excluded"}


def evaluate(args, options):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from .data import load_records, prompt_tokens

    records = load_records(args.data)
    count = min(len(records), args.limit)
    if count < 1:
        raise ValueError("No evaluation prompts")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    prompts = [prompt_tokens(records[i], tokenizer) for i in range(count)]
    if any(len(p) + args.max_new_tokens > args.max_model_len for p in prompts):
        raise ValueError("Prompt plus generation exceeds max_model_len")
    baseline = None
    if args.baseline:
        baseline = [json.loads(s) for s in Path(args.baseline).read_text().splitlines() if s.strip()]
        if len(baseline) < count or any(r.get("prompt_token_ids") != prompts[i] or r.get("target") != args.target
            or r.get("max_new_tokens") != args.max_new_tokens or r.get("mode") != "ar"
            for i, r in enumerate(baseline[:count])):
            raise ValueError("Baseline must use the same prompts, target, output budget and AR decoding")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "per_prompt.jsonl").exists():
        raise ValueError("Use a fresh output directory")
    (out / "engine_config.json").write_text(json.dumps(options, indent=2) + "\n")
    llm = LLM(**options)
    warm_count = min(args.concurrency, count)
    for _ in range(args.warmup):
        llm.generate([{"prompt_token_ids": p} for p in prompts[:warm_count]],
                     SamplingParams(temperature=0, max_tokens=min(32, args.max_new_tokens), seed=42), use_tqdm=False)
    speculative = args.mode != "ar"
    before = worker_counters(llm) if speculative else None
    if speculative:
        if any(before.values()):
            raise RuntimeError("Unexpected extra target forward during warmup")
        llm.collective_rpc("four_anchor_reset_batch_stats")
    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=42,
                            repetition_penalty=1, presence_penalty=0, frequency_penalty=0)
    started = time.perf_counter()
    # Submit the complete queue, not one prompt at a time. Native continuous
    # batching limits active requests to max_num_seqs; finished slots refill.
    responses = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=True)
    wall = time.perf_counter() - started
    if len(responses) != count:
        raise RuntimeError("Engine returned an incomplete batch")
    delta, stats = {"extra_target_forwards": 0, "bonus_page_fallbacks": 0}, {}
    if speculative:
        after = worker_counters(llm)
        delta = {k: after[k] - before[k] for k in before}
        if any(delta.values()):
            raise RuntimeError("Unexpected target forward outside native verification")
        stats = llm.collective_rpc("four_anchor_batch_stats")[0]
    rows = []
    with (out / "per_prompt.jsonl").open("x") as stream:
        for i, response in enumerate(responses):
            if list(response.prompt_token_ids) != prompts[i]:
                raise RuntimeError("Engine response order differs from submitted prompts")
            result = response.outputs[0]
            metrics = result.spec_decode_metrics
            metrics = metrics.to_dict() if metrics is not None else None
            if metrics and (metrics["num_spec_tokens"] != 15 or any(x > 15 or x < 0 for x in metrics.get("per_step_accepted", []))):
                raise RuntimeError("Invalid speculative accounting")
            if speculative and metrics is None and len(result.token_ids) > 1:
                raise RuntimeError("Speculative metrics missing")
            row = {"prompt_index": i, "mode": args.mode, "target": args.target,
                "concurrency": args.concurrency, "max_new_tokens": args.max_new_tokens,
                "prompt_token_ids": prompts[i], "token_ids": list(result.token_ids),
                "output_tokens": len(result.token_ids), "speculative_decoding": metrics}
            # Per-request times overlap: never sum them as benchmark wall time.
            timing = getattr(response, "metrics", None)
            arrival, finished = getattr(timing, "arrival_time", None), getattr(timing, "finished_time", None)
            if arrival is not None and finished is not None:
                row["latency_including_queue_seconds"] = finished - arrival
            if baseline is not None:
                row["matches_greedy_target"] = row["token_ids"] == baseline[i]["token_ids"]
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
    summary = summarize(rows, wall, args.concurrency)
    summary.update(mode=args.mode, inference_version=VERSION if speculative else "ar",
        draft_graph=False, target_graph=False, profiling_enabled=False,
        conditioning_anchors=4 if speculative else 0, speculative_tokens=15 if speculative else 0,
        query_positions={"round_one": 20, "round_two": 19 if args.mode == "two_pass" else None} if speculative else None,
        verification_block_size=16 if speculative else None,
        first_round_bonus_input="embedding_only" if speculative else None,
        first_round_target_anchor_states=4 if speculative else 0,
        second_round_anchor_states="four_first_pass_drafter_states" if args.mode == "two_pass" else None,
        anchor_policy="bonus_addon" if speculative else None,
        bonus_target_forward_required=False, parity_checked=baseline is not None,
        extra_target_forwards_measured=speculative, **delta, **stats)
    if speculative:
        summary["mean_draft_batch"] = stats["draft_requests"] / max(stats["draft_batches"], 1)
    if baseline is not None:
        summary["matching_prompts"] = sum(r["matches_greedy_target"] for r in rows)
        summary["all_match_greedy_target"] = all(r["matches_greedy_target"] for r in rows)
        # Only a matching concurrency run supports a throughput speedup claim.
        base_summary_path = Path(args.baseline).parent / "summary.json"
        if base_summary_path.exists():
            b = json.loads(base_summary_path.read_text())
            if b.get("concurrency") == args.concurrency and b.get("prompts") == count and b.get("throughput_scope") == summary["throughput_scope"]:
                summary["measured_speedup_vs_ar"] = summary["tokens_per_second_including_prefill"] / b["tokens_per_second_including_prefill"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if baseline is not None and not summary["all_match_greedy_target"]:
        raise RuntimeError("Greedy parity failed; results remain provisional")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["evaluate", "serve", "check"])
    p.add_argument("--target", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--mode", choices=["ar", "one_pass", "two_pass"], default="two_pass")
    p.add_argument("--anchor-policy", choices=["bonus_addon"], default="bonus_addon")
    p.add_argument("--proposals", type=int, choices=[15], default=15)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--memory-fraction", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--profile-steps", type=int, default=0)
    p.add_argument("--draft-graph", action="store_true")
    p.add_argument("--trace-steps", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8095)
    for name in ("data", "output", "baseline"):
        p.add_argument("--" + name)
    args = p.parse_args(argv)
    if args.mode != "ar" and not args.checkpoint:
        p.error("Speculative inference requires --checkpoint")
    if args.command == "evaluate" and (not args.data or not args.output):
        p.error("evaluate requires --data and --output")
    if min(args.limit, args.max_new_tokens) < 1 or args.warmup < 0 or not 0 < args.memory_fraction < 1:
        p.error("Invalid numeric option")
    options = engine_options(args)
    if args.command == "check":
        print(json.dumps(options, indent=2))
    elif args.command == "serve":
        command = server_command(options, args.host, args.port)
        os.execv(sys.executable, command)
    else:
        evaluate(args, options)


if __name__ == "__main__":
    main()
