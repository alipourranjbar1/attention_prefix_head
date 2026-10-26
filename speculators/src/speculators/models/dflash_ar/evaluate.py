"""Greedy verification reference using the existing cached LocalTarget adapter."""
import torch


@torch.no_grad()
def generate(target, draft, prompt, max_new_tokens=256, eos_ids=()):
    if not prompt or max_new_tokens < 1:
        raise ValueError("Need a nonempty prompt and positive token budget")
    target.reset()
    states = target.states(prompt)
    prefix = list(prompt)
    anchor = int(draft.verifier_lm_head(states.scoring[:, -1:]).argmax(-1).item())
    generated, rounds = [], []
    eos_ids = set(eos_ids)
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        if anchor in eos_ids or remaining == 1:
            generated.append(anchor)
            break
        k = min(draft.block_size, remaining - 1)
        proposals = draft.propose(states.auxiliary[:, :len(prefix)],
                                  torch.tensor([[anchor]], device=draft.device), k)[0].tolist()
        # One target call for the anchor + complete sequential draft proposal.
        checked = target.states(prefix + [anchor] + proposals)
        expected = draft.verifier_lm_head(
            checked.scoring[:, len(prefix):len(prefix) + k + 1]).argmax(-1)[0].tolist()
        accepted = 0
        for proposed, correct in zip(proposals, expected):
            if proposed != correct:
                break
            accepted += 1
        emitted = [anchor] + proposals[:accepted]
        # EOS truncates generation, not the raw verification result. Keep the
        # actual emitted count separate from the raw accepted-prefix statistic.
        for i, token in enumerate(emitted):
            if token in eos_ids:
                emitted = emitted[:i + 1]
                break
        rounds.append(dict(proposed=k, accepted=accepted, emitted=len(emitted),
                           full_block=k == draft.block_size))
        generated.extend(emitted)
        prefix.extend(emitted)
        if emitted[-1] in eos_ids:
            break
        anchor = expected[accepted]  # mismatch correction or all-accepted bonus
        states = checked
    return generated, rounds


@torch.no_grad()
def target_greedy(target, draft, prompt, max_new_tokens, eos_ids=()):
    target.reset()
    tokens, result = list(prompt), []
    for _ in range(max_new_tokens):
        states = target.states(tokens)
        token = int(draft.verifier_lm_head(states.scoring[:, -1:]).argmax(-1).item())
        tokens.append(token)
        result.append(token)
        if token in eos_ids:
            break
    return result
