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
        max_num_batched_tokens=args.max_model_len, max_num_seqs=1,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        async_scheduling=False, gpu_memory_utilization=args.memory_fraction,
        seed=42, trust_remote_code=False,
        per_request_spec_decode_metrics="detailed", generation_config="vllm",
    )
    if args.mode != "ar":
        settings = ServingSettings(str(Path(args.checkpoint).resolve()), args.proposals, args.mode)
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
    llm = LLM(**options)
    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=42,
                            repetition_penalty=1, presence_penalty=0, frequency_penalty=0)
    for _ in range(args.warmup):
        llm.generate([{"prompt_token_ids": prompts[0]}],
                     SamplingParams(temperature=0, max_tokens=min(32, args.max_new_tokens)), use_tqdm=False)
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
            if baseline is not None:
                row["matches_greedy_target"] = row["token_ids"] == baseline[i]["token_ids"]
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in {"prompt_token_ids", "token_ids"}}), flush=True)
    summary = summarize(rows)
    summary.update(mode=args.mode, conditioning_anchors=4 if args.mode != "ar" else 0,
                   speculative_tokens=args.proposals if args.mode != "ar" else 0,
                   parity_checked=baseline is not None)
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
    args = parser.parse_args(argv)
    if args.mode != "ar" and not args.checkpoint:
        parser.error("Speculative serving/evaluation requires --checkpoint")
    if args.command == "evaluate" and (not args.data or not args.output):
        parser.error("evaluate requires --data and --output")
    if args.limit < 1 or args.max_new_tokens < 1 or args.warmup < 0:
        parser.error("limit/max-new-tokens must be positive; warmup must be nonnegative")
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
