from fastapi.testclient import TestClient

from four_anchor.serve import create_app


class TinyTokenizer:
    bos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [1] + [2 + ord(c) % 50 for c in text]}

    def decode(self, tokens, **kwargs):
        return " ".join(str(t) for t in tokens)

    def apply_chat_template(self, messages, **kwargs):
        return self(" ".join(m["content"] for m in messages))["input_ids"]


def test_completion_and_chat_endpoints(components):
    _, draft, runner = components
    client = TestClient(create_app(runner, draft, TinyTokenizer()))
    assert client.get("/health").json()["new_tokens_per_pass"] == 16
    for endpoint, prompt in [("completions", {"prompt": "Hi"}),
                              ("chat/completions", {"messages": [{"role": "user", "content": "Hi"}]})]:
        response = client.post("/v1/" + endpoint, json={**prompt, "max_tokens": 5})
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["completion_tokens"] == 5
        assert response.json()["speculation"]["draft_calls"] >= 2
    assert client.post("/v1/completions", json={"prompt": "x", "temperature": 1}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": "x", "stream": True}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": "x", "top_p": 0.5}).status_code == 422
