from dataclasses import replace
import json
import random
from pathlib import Path

import httpx
import pytest
import torch
from safetensors.torch import save_file

from four_anchor.remote_teacher import FrozenTargetInterface, RemoteTeacher
from four_anchor.runtime import Decoder
from four_anchor.train import TrainConfig, TrainingTask


def make_client(tmp_path, target, runner, transform=None):
    requests = []
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "four-anchor-teacher"}]})
        payload = json.loads(request.content)
        requests.append(payload)
        ids = torch.tensor([payload["prompt"]])
        state = runner.prefill(ids)
        features = state.features[0].reshape(ids.shape[1], 2, target.config.hidden_size)
        path = tmp_path / f"request-{len(requests)}.safetensors"
        save_file({"hidden_states": features.contiguous(), "token_ids": ids[0].contiguous()}, str(path))
        generated = Decoder(runner, None).generate(ids, payload["max_tokens"], "ar", eos_ids=[])
        response = {"choices": [{"prompt_token_ids": payload["prompt"], "token_ids": generated.token_ids}],
                    "kv_transfer_params": {"hidden_states_path": str(path)}}
        if transform:
            transform(response)
        return httpx.Response(200, json=response)
    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="http://localhost/v1/", transport=transport)
    return RemoteTeacher(target, [0, 1], "http://localhost/v1", tmp_path, client=client), requests


def test_remote_features_and_teacher_tokens_match_real_qwen(components, tmp_path):
    target, _, runner = components
    client, requests = make_client(tmp_path, target, runner)
    try:
        client.check_server()
        ids = torch.tensor([[1, 2, 3, 4]])
        remote = client.prefill(ids)
        local = runner.prefill(ids)
        torch.testing.assert_close(remote.features, local.features, atol=0, rtol=0)
        retained = torch.tensor([[17, 19, 20]])
        branch = client.append(remote, retained)
        assert branch.features is None
        labels, valid = client.rollout(branch, 16)
        gold = Decoder(runner, None).generate(branch.ids, 16, "ar", eos_ids=[]).token_ids
        assert labels.tolist() == [gold]
        assert valid.sum() == 16
        assert len(requests) == 2
        assert requests[1]["prompt"] == [1, 2, 3, 4, 17, 19, 20]
        assert not list(tmp_path.glob("*.safetensors"))
    finally:
        client.close()


def test_remote_training_keeps_gradients_and_uses_two_requests(components, tmp_path):
    target, draft, runner = components
    client, requests = make_client(tmp_path, target, runner)
    try:
        cfg = TrainConfig("tiny", "data", "output", 63, [0, 1], num_draft_layers=2, dtype="float32", device="cpu")
        loss, _ = TrainingTask(draft, client, cfg, random.Random(4))(
            torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5] * 16]), torch.ones(1, 16))
        loss.backward()
        assert torch.isfinite(loss)
        assert draft.layers[0].mlp.down_proj.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in target.parameters())
        assert len(requests) == 2
    finally:
        client.close()


def test_remote_rejects_prompt_mismatch_and_removes_file(components, tmp_path):
    target, _, runner = components
    client, _ = make_client(tmp_path, target, runner,
                            lambda response: response["choices"][0].update(prompt_token_ids=[99]))
    try:
        with pytest.raises(ValueError, match="prompt token"):
            client.prefill(torch.tensor([[1, 2]]))
        assert not list(tmp_path.glob("*.safetensors"))
    finally:
        client.close()


def test_remote_rejects_text_only_response(components, tmp_path):
    target, _, runner = components
    client, _ = make_client(tmp_path, target, runner,
                            lambda response: response["choices"][0].pop("token_ids"))
    try:
        state = client.prefill(torch.tensor([[1, 2]]))
        with pytest.raises(ValueError, match="generated token_ids"):
            client.rollout(state, 16)
    finally:
        client.close()


def test_load_only_frozen_target_embeddings_and_head(components, tmp_path):
    target, _, _ = components
    target.save_pretrained(tmp_path)
    interface = FrozenTargetInterface(str(tmp_path), torch.device("cpu"), torch.float32)
    assert set(dict(interface.named_parameters())) == {"embed_tokens.weight", "lm_head.weight"}
    assert all(not p.requires_grad for p in interface.parameters())
    torch.testing.assert_close(interface.embed_tokens.weight, target.get_input_embeddings().weight)
    torch.testing.assert_close(interface.lm_head.weight, target.get_output_embeddings().weight)


def test_remote_never_deletes_an_external_path(components, tmp_path):
    target, _, runner = components
    client, _ = make_client(tmp_path, target, runner)
    try:
        with pytest.raises(ValueError, match="inside this run"):
            client._handle({"kv_transfer_params": {"hidden_states_path": str(tmp_path.parent / "unrelated.safetensors")}})
    finally:
        client.close()
