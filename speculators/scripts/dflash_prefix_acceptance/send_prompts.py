#!/usr/bin/env python3
"""Sequential greedy requests for collection, output parity and timing checks."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8213")
    p.add_argument("--model", default="qwen3-4b-prefix-dflash")
    p.add_argument("--prompts", type=Path, required=True,
                   help='JSONL, one {"prompt": "..."} object per line')
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output, args.prompts.open() as prompts:
        for index, line in enumerate(prompts):
            if args.limit and index >= args.limit:
                break
            row = json.loads(line)
            payload = dict(model=args.model, prompt=row["prompt"], temperature=0,
                           max_tokens=args.max_tokens, stream=False, seed=42)
            request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=3600) as response:
                result = json.load(response)
            elapsed = time.perf_counter() - start
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            record = dict(index=index, prompt=row["prompt"], seconds=elapsed,
                          completion_tokens=tokens, tokens_per_second=tokens / elapsed,
                          text=result["choices"][0]["text"])
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(json.dumps({k: v for k, v in record.items() if k not in {"text", "prompt"}}), flush=True)


if __name__ == "__main__":
    main()
