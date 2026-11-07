"""Launch native vLLM serving or a local, sequential GSM8K prompt benchmark."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

from .vllm_support import ServingSettings


def engine_options(args):
    options = dict(
        model=args.target, dtype="bfloat16", tensor_parallel_size=1,
        enforce_eager=True, max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len, max_num_seqs=1, block_size=128,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        async_scheduling=False, gpu_memory_utilization=args.memory_fraction,
        seed=42, trust_remote_code=False,
        per_request_spec_decode_metrics=("none" if args.mode == "ar" else "detailed"), generation_config="vllm",
    )
    if args.mode != "ar":
        settings = ServingSettings(str(Path(args.checkpoint).resolve()), args.proposals, args.mode,
                                   trace_steps=getattr(args, "trace_steps", False),
                                   profile_steps=getattr(args, "profile_steps", 0),
                                   draft_graph=getattr(args, "draft_graph", False),
                                   anchor_policy=getattr(args, "anchor_policy", "trained"))
        settings.metadata()
        options.update(
            worker_cls="four_anchor.vllm_ascend.FourAnchorWorker",
            speculative_config={"method": "custom_class",
                                "model": "four_anchor.vllm_ascend.FourAnchorProposer",
                                "num_speculative_tokens": args.proposals},
            # Ascend's additional_config schema rejects foreign extension
            # keys. A namespaced HF override is serialized with the engine
            # config and does not alter any target architecture parameter.
            hf_overrides={"four_anchor_serving": asdict(settings)},
        )
    return options


def server_command(options, host, port):
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    for key, value in options.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            command.append("--" + ("" if value else "no-") + flag)
        else:
            command += ["--" + flag, json.dumps(value) if isinstance(value, dict) else str(value)]
    return command + ["--host", host, "--port", str(port), "--served-model-name", "four-anchor-qwen3"]


def summarize(rows):
    tokens = sum(r["output_tokens"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    cycles = sum((r.get("speculative_decoding") or {}).get("num_spec_steps", 0) for r in rows)
    accepted = sum((r.get("speculative_decoding") or {}).get("num_accepted_draft_tokens", 0) for r in rows)
    return {"prompts": len(rows), "output_tokens": tokens, "seconds": seconds,
            "tokens_per_second_including_prefill": tokens / max(seconds, 1e-9),
            "verification_cycles": cycles,
            "accepted_draft_per_verification": accepted / cycles if cycles else None,
            "acceptance_length_including_bonus": 1 + accepted / cycles if cycles else None,
            "acceptance_note": "vLLM verification accounting, before EOS/output-budget truncation; anchors excluded"}


def worker_counters(llm):
    values = llm.collective_rpc("four_anchor_counters")
    if len(values) != 1 or set(values[0]) != {"extra_target_forwards", "bonus_page_fallbacks"}:
        raise RuntimeError("Expected counters from exactly one four-anchor worker")
    if any(type(v) is not int or v < 0 for v in values[0].values()):
        raise RuntimeError("Invalid target-forward counters")
    return values[0]


def evaluate(args, options):
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from vllm import LLM, SamplingParams
    from .data import load_records, prompt_tokens

    records = load_records(args.data)
    count = min(len(records), args.limit)
    if count < 1:
        raise ValueError("Evaluation requires at least one local prompt")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    prompts = [prompt_tokens(records[i], tokenizer) for i in range(count)]
    if any(len(p) + args.max_new_tokens > args.max_model_len for p in prompts):
        raise ValueError("A prompt plus max_new_tokens exceeds max_model_len")
    baseline = None
    if args.baseline:
        baseline = [json.loads(line) for line in Path(args.baseline).read_text().splitlines() if line.strip()]
        if len(baseline) < count or any(baseline[i]["prompt_token_ids"] != prompts[i] for i in range(count)):
            raise ValueError("Baseline must contain the identical prompts in the same order")
        if any(r["target"] != args.target or r["max_new_tokens"] != args.max_new_tokens or r["mode"] != "ar"
               for r in baseline[:count]):
            raise ValueError("Baseline target, output budget and mode must match")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "per_prompt.jsonl").exists():
        raise ValueError("Use a fresh OUTPUT_DIR; per_prompt.jsonl already exists")
    (output / "engine_config.json").write_text(json.dumps(options, indent=2) + "\n")
    version = (ServingSettings(args.checkpoint, args.proposals, args.mode,
                              anchor_policy=getattr(args, "anchor_policy", "trained")).inference_version()
               if args.mode != "ar" else "ar")
    llm = LLM(**options)
    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=42,
                            repetition_penalty=1, presence_penalty=0, frequency_penalty=0)
    for _ in range(args.warmup):
        llm.generate([{"prompt_token_ids": prompts[0]}],
                     SamplingParams(temperature=0, max_tokens=min(32, args.max_new_tokens)), use_tqdm=False)
    counted_versions = {"target_four_materialized_v3", "accepted_prefix19_v1", "bonus_addon20_draft4_v1"}
    no_extra_versions = {"accepted_prefix19_v1", "bonus_addon20_draft4_v1"}
    counters = worker_counters(llm) if version in counted_versions else None
    if version in no_extra_versions and any(counters.values()):
        raise RuntimeError("Cached inference unexpectedly made an extra target forward during warmup")
    rows = []
    with (output / "per_prompt.jsonl").open("x") as stream:
        for i, prompt in enumerate(tqdm(prompts, desc=f"vLLM {args.mode}", unit="prompt")):
            started = time.perf_counter()
            response = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0]
            seconds = time.perf_counter() - started
            result = response.outputs[0]
            metrics = result.spec_decode_metrics
            metrics = metrics.to_dict() if metrics is not None else None
            if metrics and (metrics["num_spec_tokens"] != args.proposals or
                            any(j > args.proposals for j in metrics.get("per_step_accepted", []))):
                raise RuntimeError("Acceptance accounting exceeds the configured proposal window")
            if args.mode != "ar" and metrics is None and len(result.token_ids) > 1:
                raise RuntimeError("vLLM returned no speculative metrics; do not report this as a speculative run")
            row = {"prompt_index": i, "mode": args.mode, "target": args.target,
                   "max_new_tokens": args.max_new_tokens, "prompt_token_ids": prompt,
                   "token_ids": list(result.token_ids), "output_tokens": len(result.token_ids),
                   "seconds": seconds, "tokens_per_second": len(result.token_ids) / max(seconds, 1e-9),
                   "speculative_decoding": metrics}
            if counters is not None:
                after = worker_counters(llm)  # after the timed generate call
                delta = {key: after[key] - counters[key] for key in counters}
                if any(v < 0 for v in delta.values()):
                    raise RuntimeError("Worker counters reset during evaluation")
                if version in no_extra_versions and any(delta.values()):
                    raise RuntimeError("Cached inference unexpectedly made an extra target forward")
                row.update(delta)
                counters = after
            if baseline is not None:
                row["matches_greedy_target"] = row["token_ids"] == baseline[i]["token_ids"]
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in {"prompt_token_ids", "token_ids"}}), flush=True)
    summary = summarize(rows)
    summary.update(draft_graph=getattr(args, "draft_graph", False) and args.mode != "ar", inference_version=version,
                   extra_target_forwards=sum(r.get("extra_target_forwards", 0) for r in rows),
                   bonus_page_fallbacks=sum(r.get("bonus_page_fallbacks", 0) for r in rows),
                   profiling_enabled=getattr(args, "profile_steps", 0) > 0,
                   mode=args.mode, conditioning_anchors=4 if args.mode != "ar" else 0,
                   speculative_tokens=args.proposals if args.mode != "ar" else 0,
                   parity_checked=baseline is not None)
    if version == "target_four_materialized_v3":
        if summary["extra_target_forwards"] < summary["verification_cycles"]:
            raise RuntimeError("Target-state drafting is missing its bonus-forward accounting")
        summary["acceptance_note"] += "; bonus-materialization forwards are additional and excluded from verification-cycle denominator"
        summary["query_positions"] = 19
        summary["first_round_target_anchor_states"] = 4
    if version == "accepted_prefix19_v1":
        summary.update(query_positions=19, first_round_target_anchor_states=4,
                       anchor_policy="accepted_prefix", bonus_target_forward_required=False,
                       first_round_anchor_end="last_processed_token",
                       redraft_generated_anchor_features="first_pass_drafter",
                       extra_target_forwards_measured=True)
    if version == "bonus_addon20_draft4_v1":
        summary.update(query_positions={"round_one": 20, "round_two": 19},
                       verification_block_size=16, first_round_target_anchor_states=4,
                       first_round_bonus_input="embedding_only", anchor_policy="bonus_addon",
                       second_round_anchor_states="four_first_pass_drafter_states",
                       second_round_extra_bonus_slots=0, bonus_target_forward_required=False,
                       first_round_anchor_end="last_processed_token",
                       extra_target_forwards_measured=True)
    if baseline is not None:
        summary["matching_prompts"] = sum(r["matches_greedy_target"] for r in rows)
        summary["all_match_greedy_target"] = all(r["matches_greedy_target"] for r in rows)
        summary["measured_speedup_vs_ar"] = (
            summary["tokens_per_second_including_prefill"] /
            summarize(baseline[:count])["tokens_per_second_including_prefill"]
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if baseline is not None and not summary["all_match_greedy_target"]:
        raise RuntimeError("Greedy parity failed. Results are provisional; see per_prompt.jsonl")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve", "evaluate", "check"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=["ar", "one_pass", "two_pass"], default="two_pass")
    parser.add_argument("--proposals", type=int, default=15)
    parser.add_argument("--anchor-policy", choices=["trained", "accepted_prefix", "bonus_addon"], default="trained")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--baseline")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--draft-graph", action="store_true")
    parser.add_argument("--trace-steps", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode != "ar" and not args.checkpoint:
        parser.error("Speculative serving/evaluation requires --checkpoint")
    if args.command == "evaluate" and (not args.data or not args.output):
        parser.error("evaluate requires --data and --output")
    if args.limit < 1 or args.max_new_tokens < 1 or args.warmup < 0 or args.profile_steps < 0:
        parser.error("limit/max-new-tokens must be positive; warmup/profile-steps must be nonnegative")
    options = engine_options(args)
    if args.command == "check":
        print(json.dumps(options, indent=2))
    elif args.command == "serve":
        command = server_command(options, args.host, args.port)
        print(json.dumps({"event": "launch_vllm", "argv": command}), flush=True)
        os.execv(sys.executable, command)
    else:
        evaluate(args, options)


if __name__ == "__main__":
    main()
