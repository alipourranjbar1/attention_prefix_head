#!/usr/bin/env python3
"""Create explicit experiment configs without editing the trained checkpoint."""
import argparse
import json
import shlex
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    draft, target = args.draft.resolve(strict=True), args.target.resolve(strict=True)
    cfg = json.loads((draft / "config.json").read_text())
    json.loads((target / "config.json").read_text())
    if cfg.get("prefix_selector_kind") != "compiled_prefix_v3":
        p.error("This patch requires a compiled_prefix_v3 checkpoint")
    k = cfg["block_size"] - 1
    if not 1 <= k <= 15:
        p.error("Launcher supports 1..15 proposals")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    lengths = sorted(set([x for x in (1, 4, 8) if x <= k] + [k]))
    for mode in ("collect", "shadow", "adaptive"):
        config = dict(mode=mode, verify_lengths=lengths,
                      trace_dir=str(out / ("traces" if mode == "collect" else "shadow_traces")),
                      head_path=str(out / "head/best.pt"))
        (out / f"{mode}.json").write_text(json.dumps(config, indent=2) + "\n")
    root = Path(__file__).resolve().parents[3]
    values = dict(SPEC_MAIN=str(root), DRAFT_MODEL=str(draft), TARGET_MODEL=str(target),
                  NUM_SPECULATIVE_TOKENS=str(k), ACCEPTANCE_EXPERIMENT=str(out),
                  DRAFT_ENFORCE_EAGER="1")
    (out / "env.sh").write_text("\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in values.items()) + "\n")
    print(f"Created configs in {out}; K={k}, supported verification lengths={lengths}")


if __name__ == "__main__":
    main()
