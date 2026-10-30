"""Validate the live vLLM contract before launching distributed training."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, GenerationConfig

from .data import load_records
from .remote_teacher import RemoteTeacher


def check(config):
    target_config = AutoConfig.from_pretrained(config["target"], local_files_only=True)
    # The contract check only needs metadata and a tiny device/dtype marker.
    target = SimpleNamespace(config=target_config, get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0)))
    client = RemoteTeacher(target, config["target_layer_ids"], config["vllm_endpoint"],
                           config["hidden_states_dir"], config["teacher_model_name"],
                           config["request_timeout"], config["max_retries"])
    try:
        client.check_server()
        data = load_records(config["data"], config["split"])
        ids = list(data[0]["input_ids"])[:min(16, config["max_sequence_length"])]
        if not ids:
            raise ValueError("First dataset row has no tokens")
        state = client.prefill(torch.tensor([ids], dtype=torch.long))
        try:
            generation = GenerationConfig.from_pretrained(config["target"], local_files_only=True)
        except OSError:
            generation = GenerationConfig.from_model_config(target_config)
        eos = generation.eos_token_id
        eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
        labels, valid = client.rollout(state, config["num_proposals"], eos)
        result = {"status": "passed", "hidden_states_shape": list(state.features.shape),
                  "generated_token_slots": labels.shape[1], "valid_generated_tokens": int(valid.sum()),
                  "target_calls": client.calls}
        print(json.dumps(result), flush=True)
        Path(config["output"]).parent.joinpath("teacher_contract.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    check(json.loads(Path(parser.parse_args().config).read_text()))
