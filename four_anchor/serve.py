"""Single-device, serialized, non-streaming OpenAI-style research server."""
from threading import Lock
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .data import prompt_tokens
from .runtime import Decoder


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "four-anchor-dflash"
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = 0
    stream: bool = False
    n: int = 1


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "four-anchor-dflash"
    messages: list[dict[str, str]]
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = 0
    stream: bool = False
    n: int = 1


def create_app(runner, drafter, tokenizer):
    app = FastAPI(title="Four-anchor DFlash reference server")
    lock = Lock()
    decoder = Decoder(runner, drafter)
    device = next(runner.target.parameters()).device

    def complete(request, row, chat=False):
        if request.temperature != 0 or request.stream or request.n != 1:
            raise HTTPException(400, "Supported: temperature=0, stream=false, n=1")
        if request.model != "four-anchor-dflash":
            raise HTTPException(404, "Unknown model; see /v1/models")
        try:
            ids = prompt_tokens(row, tokenizer)
            if len(ids) + request.max_tokens > runner.target.config.max_position_embeddings:
                raise ValueError("Prompt plus requested completion exceeds target context window")
            # Target hook capture and target cache are request-local only under this lock.
            with lock:
                out = decoder.generate(torch.tensor([ids], device=device), request.max_tokens)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        text = tokenizer.decode(out.token_ids, skip_special_tokens=True)
        eos = runner.target.generation_config.eos_token_id
        eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
        choice = {"index": 0, "finish_reason": "stop" if out.token_ids and out.token_ids[-1] in eos else "length"}
        choice.update({"message": {"role": "assistant", "content": text}} if chat else {"text": text})
        return {"id": "cmpl-" + uuid.uuid4().hex, "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()), "model": request.model, "choices": [choice],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out.token_ids),
                          "total_tokens": len(ids) + len(out.token_ids)}, "speculation": out.metrics()}

    @app.get("/health")
    def health():
        return {"status": "ok", "anchors": drafter.cfg.num_anchors,
                "new_tokens_per_pass": drafter.cfg.num_proposals, "draft_passes": 2}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "four-anchor-dflash", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/completions")
    def completion(request: CompletionRequest):
        return complete(request, {"prompt": request.prompt})

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        return complete(request, {"messages": request.messages}, chat=True)

    return app
