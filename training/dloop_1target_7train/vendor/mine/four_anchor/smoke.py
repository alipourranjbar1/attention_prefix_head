"""Offline CPU smoke: real tiny Qwen3, teacher data, train, resume, eval, HTTP.

Creates random tiny weights. The result tests plumbing, not model quality.
Run: python -m four_anchor.smoke --output /tmp/dflash4a-smoke
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from .cli import main as cli
from .train import TrainConfig


def smoke(output):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty smoke output directory")
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "tiny_target"
    vocab = {"[PAD]": 0, "[UNK]": 1, **{f"t{i}": i for i in range(2, 63)}, "[MASK]": 63}
    token_backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    token_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=token_backend, pad_token="[PAD]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        max_position_embeddings=512, eos_token_id=None, pad_token_id=0)
    target = Qwen3ForCausalLM(config)
    target.save_pretrained(target_path)
    tokenizer.save_pretrained(target_path)
    prompts = output / "train_prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in ["t2 t3 t4 t5", "t6 t7 t8", "t9 t10", "t11"]))
    heldout = output / "heldout_prompts.jsonl"
    heldout.write_text('{"prompt":"t20 t21 t22 t23"}\n{"prompt":"t31 t32"}\n')
    data = output / "teacher.jsonl"
    cli(["prepare", "--target", str(target_path), "--data", str(prompts), "--output", str(data),
         "--max-new-tokens", "32"])
    cfg = TrainConfig(str(target_path), str(data), str(output / "train"), 63, [0, 1],
                      device="cpu", dtype="float32", num_draft_layers=2, steps=2,
                      max_sequence_length=64, gradient_accumulation=1, warmup_steps=0,
                      checkpoint_every=1, boundary_exploration=1)
    config_path = output / "train_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2))
    cli(["train", "--config", str(config_path)])
    checkpoint = output / "train" / "checkpoint-000002"
    # Resume step 1 using the same total schedule. The final checkpoint must match
    # the uninterrupted run bit-for-bit on CPU, including bridge parameters.
    cli(["train", "--config", str(config_path), "--resume", str(output / "train" / "checkpoint-000001"),
         "--output", str(output / "resumed")])
    expected = load_file(str(checkpoint / "model.safetensors"))
    actual = load_file(str(output / "resumed" / "checkpoint-000002" / "model.safetensors"))
    identical = all(torch.equal(expected[k], actual[k]) for k in expected)
    if not identical:
        raise AssertionError("Resumed weights differ from uninterrupted training")
    cli(["generate", "--checkpoint", str(checkpoint), "--prompt", "t20 t21 t22 t23", "--max-new-tokens", "20",
         "--output", str(output / "generation.json")])
    cli(["evaluate", "--checkpoint", str(checkpoint), "--data", str(heldout), "--output", str(output / "evaluation"),
         "--max-new-tokens", "24", "--warmup", "1", "--limit", "2"])
    from fastapi.testclient import TestClient
    from .model import FourAnchorDFlash
    from .runtime import load_target, TargetRunner
    from .serve import create_app
    target, tokenizer = load_target(str(target_path))
    drafter = FourAnchorDFlash.load(checkpoint, target)
    runner = TargetRunner(target, [0, 1])
    try:
        client = TestClient(create_app(runner, drafter, tokenizer))
        response = client.post("/v1/completions", json={"prompt": "t20 t21 t22 t23", "max_tokens": 5})
        assert response.status_code == 200, response.text
    finally:
        runner.close()
    summary = {"resume_bitwise_equal": identical, "http_status": response.status_code,
               "checkpoint": str(checkpoint), "torch": torch.__version__, "device": "cpu",
               "evaluation": json.loads((output / "evaluation" / "summary.json").read_text()),
               "note": "Random tiny target, two optimizer steps: correctness smoke only; no speedup claim."}
    (output / "smoke_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    smoke(parser.parse_args().output)
