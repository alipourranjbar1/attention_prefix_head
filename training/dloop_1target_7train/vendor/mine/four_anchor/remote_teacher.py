"""Training-only client for the repository's vLLM hidden-state server.

The trainer loads only frozen embedding/head weights, not a target transformer.
Server capture IDs are decoder indices + 1 (vLLM includes embedding slot zero).
Launch with --no-include-last-layer: each payload contains exactly these taps.
"""
from dataclasses import dataclass
from pathlib import Path
import time

import httpx
import torch
from torch import nn
from transformers import AutoConfig, GenerationConfig
from hs_connectors import FileTransfer
from hs_connectors.transfer import wait_for_lock
from speculators.utils.loading import load_model_layers


class FrozenTargetInterface(nn.Module):
    def __init__(self, model_path, device, dtype, include_final=False):
        super().__init__()
        if not Path(model_path).is_dir():
            raise ValueError("Remote teacher training requires a local target checkpoint")
        self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if self.config.model_type != "qwen3":
            raise ValueError("Remote training currently supports Qwen3 only")
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
        except OSError:
            self.generation_config = GenerationConfig.from_model_config(self.config)
        keys = ["model.embed_tokens.weight", "lm_head.weight"]
        if include_final:
            keys.append("model.norm.weight")
        weights = load_model_layers(keys, model_path)
        embedding = weights["model.embed_tokens.weight"].to(device=device, dtype=dtype)
        head = weights.get("lm_head.weight")
        if head is None and not self.config.tie_word_embeddings:
            raise ValueError("Untied target checkpoint is missing lm_head.weight")
        self.embed_tokens = nn.Embedding.from_pretrained(embedding, freeze=True,
                                                       padding_idx=self.config.pad_token_id)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False, device="meta")
        head = embedding if head is None else head.to(device=device, dtype=dtype)
        self.lm_head.weight = nn.Parameter(head, requires_grad=False)
        if include_final:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
            self.final_norm = Qwen3RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            self.final_norm.weight = nn.Parameter(weights["model.norm.weight"].to(device=device, dtype=dtype),
                                                  requires_grad=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


@dataclass
class RemoteState:
    ids: torch.Tensor
    features: torch.Tensor | None
    final_hidden: torch.Tensor | None = None


class RemoteTeacher:
    def __init__(self, target, layer_ids, endpoint, hidden_states_dir, model_name="four-anchor-teacher",
                 timeout=180, max_retries=8, client=None, include_final=False):
        self.target = target
        self.layer_ids = list(layer_ids)
        self.include_final = include_final
        self.device = target.get_input_embeddings().weight.device
        self.dtype = target.get_input_embeddings().weight.dtype
        self.storage = Path(hidden_states_dir).resolve()
        self.transfer = FileTransfer(self.storage)
        self.model_name, self.max_retries = model_name, max_retries
        self.client = client or httpx.Client(base_url=endpoint.rstrip("/") + "/", timeout=timeout, trust_env=False)
        self.calls = 0

    def close(self):
        self.client.close()

    def check_server(self):
        response = self.client.get("models")
        response.raise_for_status()
        models = [m["id"] for m in response.json()["data"]]
        if self.model_name not in models:
            raise ValueError(f"Teacher model {self.model_name!r} not served; available: {models}")

    def _request(self, ids, count):
        payload = {"model": self.model_name, "prompt": ids[0].tolist(), "max_tokens": count,
                   "temperature": 0, "top_p": 1.0, "top_k": -1,
                   "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                   "return_token_ids": True}
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                response = self.client.post("completions", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as error:
                permanent = isinstance(error, httpx.HTTPStatusError) and 400 <= error.response.status_code < 500
                if permanent or attempt == self.max_retries:
                    raise
                time.sleep(min(0.5 * 2 ** attempt, 5.0))

    def _handle(self, response):
        params = response.get("kv_transfer_params") or {}
        value = params.get("hidden_states_path") or params.get("handle")
        if not isinstance(value, str):
            raise ValueError("Teacher response has no hidden-state file handle")
        path = Path(value).resolve()
        if not path.is_relative_to(self.storage) or path.suffix != ".safetensors":
            raise ValueError("Teacher file must be a safetensors file inside this run's hidden-state directory")
        return path

    @staticmethod
    def _choice(response, ids):
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("prompt_token_ids") != ids[0].tolist():
            raise ValueError("Teacher prompt token IDs do not match the requested prefix")
        return choices[0]

    def _delete(self, path):
        lock = Path(str(path) + ".lock")
        if lock.exists():
            wait_for_lock(str(lock), timeout=30)
        path.unlink(missing_ok=True)

    def prefill(self, ids):
        response = self._request(ids, 1)
        handle = self._handle(response)
        try:
            self._choice(response, ids)
            data = self.transfer.get_generated(str(handle))
            if data is None or data["token_ids"].tolist() != ids[0].tolist():
                raise ValueError("Teacher hidden-state payload is absent or has different token IDs")
            features = data["hidden_states"]
            # The server's include-last-layer option appends the final decoder
            # output only when it is not already in the requested capture list.
            final_id = self.target.config.num_hidden_layers - 1
            capture_ids = self.layer_ids + ([final_id] if self.include_final and final_id not in self.layer_ids else [])
            expected = (ids.shape[1], len(capture_ids), self.target.config.hidden_size)
            if tuple(features.shape) != expected or not torch.isfinite(features).all():
                raise ValueError(f"Expected finite hidden states shaped {expected}, got {tuple(features.shape)}")
            final = features[:, capture_ids.index(final_id)][None].to(device=self.device, dtype=self.dtype) if self.include_final else None
            features = features[:, :len(self.layer_ids)].flatten(1)[None].to(device=self.device, dtype=self.dtype)
            return RemoteState(ids, features, final)
        finally:
            self._delete(handle)

    def append(self, state, ids):
        # No target features from proposed tokens are ever supplied to the drafter.
        return RemoteState(torch.cat([state.ids, ids], dim=1), None)

    def rollout(self, state, count, eos_ids=()):
        response = self._request(state.ids, count)
        handle = self._handle(response)
        try:
            choice = self._choice(response, state.ids)
            tokens = choice.get("token_ids")
            if not isinstance(tokens, list) or not 0 < len(tokens) <= count:
                raise ValueError("Teacher must return generated token_ids; text re-tokenization is not allowed")
            if any(not isinstance(t, int) or t < 0 or t >= self.target.config.vocab_size for t in tokens):
                raise ValueError("Teacher returned invalid token IDs")
            eos_at = next((i for i, t in enumerate(tokens) if t in eos_ids), None)
            if eos_at is not None:
                tokens = tokens[:eos_at + 1]
            if len(tokens) < count and eos_at is None:
                raise ValueError("Teacher returned a short continuation without EOS")
            padding = count - len(tokens)
            return (torch.tensor([tokens + [0] * padding], device=self.device, dtype=torch.long),
                    torch.tensor([[1.] * len(tokens) + [0.] * padding], device=self.device))
        finally:
            # vLLM also writes prompt states for rollout calls. Wait for its writer
            # and remove this run's file; do not keep or feed these states to pass two.
            self._delete(handle)
