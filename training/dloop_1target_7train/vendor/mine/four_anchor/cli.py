import argparse
from dataclasses import asdict
from pathlib import Path
import json

import torch

from .data import prepare_data, prompt_tokens
from .model import FourAnchorDFlash
from .provenance import record
from .runtime import Decoder, TargetRunner, load_target


def main(argv=None):
    parser = argparse.ArgumentParser(description="Four-anchor regular DFlash: data, train, decode, evaluate, serve")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="Train from a JSON config (optionally with torchrun)")
    train_parser.add_argument("--config", required=True)
    for key in ["target", "data", "output", "device", "dtype", "resume", "initialize-dflash"]:
        train_parser.add_argument("--" + key)
    train_parser.add_argument("--steps", type=int)
    for command in ["prepare", "generate", "evaluate", "serve"]:
        p = sub.add_parser(command)
        p.add_argument("--target", help="Optional checkpoint target-path override; must be the same target weights")
        p.add_argument("--revision")
        p.add_argument("--device", default="cpu")
        p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
        p.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
        if command != "prepare":
            p.add_argument("--checkpoint", required=True)
        if command in {"prepare", "evaluate"}:
            p.add_argument("--data", required=True)
            p.add_argument("--output", required=True)
            p.add_argument("--limit", type=int, default=100 if command == "evaluate" else None)
        if command != "serve":
            p.add_argument("--max-new-tokens", type=int, default=256)
        if command == "prepare":
            p.add_argument("--max-prompt-tokens", type=int, default=1536)
        if command == "generate":
            p.add_argument("--prompt", required=True)
            p.add_argument("--mode", choices=["ar", "one_pass", "two_pass"], default="two_pass")
            p.add_argument("--output", help="Optional JSON output file including token IDs and cycle trace")
        if command == "evaluate":
            p.add_argument("--warmup", type=int, default=2)
        if command == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--provenance-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        from .train import TrainConfig, train
        settings = json.loads(Path(args.config).read_text())
        settings.update({k: v for k, v in vars(args).items() if k not in {"command", "config"} and v is not None})
        print(train(TrainConfig(**settings)))
        return
    metadata = None
    if args.command != "prepare":
        metadata = json.loads((Path(args.checkpoint) / "four_anchor_config.json").read_text())["draft_config"]
    target_name = args.target or (metadata and metadata["target_name"])
    if not target_name:
        parser.error("prepare requires --target")
    revision = args.revision or (metadata and metadata.get("target_revision"))
    target, tokenizer = load_target(target_name, args.device, args.dtype, revision, args.attention)
    if args.command == "prepare":
        record(Path(args.output).parent / (Path(args.output).stem + "_provenance"), "prepare", vars(args))
        prepare_data(target, tokenizer, args.data, args.output, args.max_new_tokens, args.max_prompt_tokens, args.limit)
        return
    drafter = FourAnchorDFlash.load(args.checkpoint, target).eval()
    runner = TargetRunner(target, drafter.cfg.target_layer_ids)
    try:
        if args.command == "generate":
            ids = prompt_tokens({"prompt": args.prompt}, tokenizer)
            result = Decoder(runner, drafter).generate(torch.tensor([ids], device=args.device), args.max_new_tokens, args.mode)
            row = {"text": tokenizer.decode(result.token_ids, skip_special_tokens=True),
                   "token_ids": result.token_ids, "metrics": result.metrics(), "cycles": [asdict(c) for c in result.cycles]}
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps(row, indent=2))
        elif args.command == "evaluate":
            from .evaluate import evaluate
            print(json.dumps(evaluate(runner, drafter, tokenizer, args.data, args.output, args.limit,
                                      args.max_new_tokens, args.warmup, args.checkpoint), indent=2))
        else:
            import uvicorn
            from .serve import create_app
            record(args.provenance_dir, "serve", vars(args))
            uvicorn.run(create_app(runner, drafter, tokenizer), host=args.host, port=args.port, workers=1)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
