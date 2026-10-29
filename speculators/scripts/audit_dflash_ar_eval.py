"""Diagnose AR proposal parity; this is not an acceptance/speed benchmark.

Replay each cached proposal through the training backbone, conditioned on the
same prefix and preceding proposal tokens. Optionally compare the HF target
features with the hidden-state server used for training.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch


@torch.no_grad()
def replay_training_forward(draft, features, anchor, proposals):
    """Pin one anchor and replay the actual training code on identical inputs."""
    prefix = features.shape[1]
    block = draft.block_size
    count = proposals.shape[1]
    length = prefix + block + 1
    device = features.device
    ids = torch.zeros((1, length), dtype=torch.long, device=device)
    ids[:, prefix:prefix + 1] = anchor
    ids[:, prefix + 1:prefix + count + 1] = proposals
    hidden = features.new_zeros((1, length, features.shape[-1]))
    hidden[:, :prefix] = features
    documents = torch.zeros_like(ids)
    anchors = torch.tensor([prefix], device=device)
    full = (draft._create_attention_mask(documents, length, anchors, device)
            if draft.uses_full_attn else None)
    sliding = (draft._create_attention_mask(documents, length, anchors, device,
                                          draft.sliding_window)
               if draft.uses_sliding_window_attn else None)
    masks = (full, sliding, anchors, torch.tensor([True], device=device))
    with patch.object(draft, '_build_attention_mask', return_value=masks):
        logits = draft._backbone_forward(
            hidden_states=hidden, input_ids=ids,
            loss_mask=torch.ones_like(ids, dtype=features.dtype),
            verifier_last_hidden_states=features.new_zeros((1, length, draft.hidden_size)),
            document_ids=documents, max_anchors=1,
        )[1]
    return logits[:, :count]


def difference(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {a.shape} versus {b.shape}')
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return dict(shape=list(a.shape), finite=finite,
                max_abs=float((a - b).abs().max()) if finite else None,
                relative_l2=float(torch.linalg.vector_norm(a - b) /
                                  torch.linalg.vector_norm(a).clamp_min(1e-12)) if finite else None)


def check_checkpoint_keys(draft, directory):
    from speculators.utils.loading import list_checkpoint_keys
    stored = set(list_checkpoint_keys(directory))
    expected = set(draft.state_dict())
    reconstructed = {'embed_tokens.weight', 'lm_head.weight', 'verifier_lm_head.weight',
                     'verifier_norm.weight', 't2d', 'd2t'}
    missing = expected - stored - reconstructed
    unexpected = stored - expected
    if missing or unexpected:
        raise ValueError(f'Checkpoint key mismatch: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}')
    return dict(stored_tensors=len(stored), required_learned_tensors=len(expected - reconstructed))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    from speculators.models.dflash_ar import DFlashARDraftModel, DFlashARSpeculatorConfig
    from speculators.models.dflash_ar.evaluate import generate, target_greedy
    from speculators.models.retrace.target import LocalTarget, RemoteTarget
    from speculators.train.utils import save_train_command

    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--draft', required=True)
    p.add_argument('--target', help='Defaults to the target recorded in the checkpoint')
    p.add_argument('--prompts', required=True, help='Same JSONL file as evaluate_dflash_ar.py')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--num-prompts', type=int, default=3)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--vllm-endpoint', help='Optional training hidden-state endpoint, e.g. http://127.0.0.1:8080/v1')
    p.add_argument('--target-attn', choices=['sdpa', 'eager'], default='sdpa')
    p.add_argument('--draft-attn', choices=['sdpa', 'eager'], default='sdpa')
    a = p.parse_args()
    if min(a.rounds, a.num_prompts, a.max_new_tokens) < 1:
        p.error('Round, prompt, and token limits must be positive')
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        torch.npu.set_device(a.device)
    dtype = torch.float32 if a.device == 'cpu' else torch.bfloat16
    cfg = DFlashARSpeculatorConfig.from_pretrained(a.draft)
    recorded_target = cfg.speculators_config.verifier.name_or_path
    target_path = a.target or recorded_target
    if not target_path:
        p.error('No target path is recorded; supply --target')
    cfg.transformer_layer_config._attn_implementation = a.draft_attn
    # The loader uses the target recorded in config, not its verifier= argument.
    cfg.speculators_config.verifier.name_or_path = target_path
    draft = DFlashARDraftModel.from_pretrained(a.draft, config=cfg, torch_dtype=dtype).to(a.device).eval()
    keys = check_checkpoint_keys(draft, a.draft)
    model = AutoModelForCausalLM.from_pretrained(
        target_path, torch_dtype=dtype, attn_implementation=a.target_attn).to(a.device).eval()
    draft.embed_tokens = model.model.embed_tokens
    draft.lm_head = draft.verifier_lm_head = model.lm_head
    draft.verifier_norm = model.model.norm
    if min(draft.target_layer_ids) < 0 or max(draft.target_layer_ids) >= model.config.num_hidden_layers:
        raise ValueError('Auxiliary layers must precede the final target norm')
    if a.vllm_endpoint and draft.target_layer_ids != sorted(set(draft.target_layer_ids)):
        raise ValueError('vLLM returns hidden layers in model order; layer IDs must be unique and sorted')
    target = LocalTarget(model, draft.target_layer_ids)
    remote = (RemoteTarget(draft, a.vllm_endpoint, target_path, len(draft.target_layer_ids))
              if a.vllm_endpoint else None)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    save_train_command(output)
    events, requests = [], []
    original = draft.propose
    current_request = 0

    @torch.no_grad()
    def audited(features, anchor, count=None):
        if len(events) >= a.rounds:
            return original(features, anchor, count)
        captured = []
        handle = draft.lm_head.register_forward_hook(
            lambda module, inputs, value: captured.append(value.detach().float().cpu()))
        try:
            proposals = original(features, anchor, count)
        finally:
            handle.remove()
        cached_logits = torch.cat(captured, dim=1)
        full_logits = replay_training_forward(draft, features, anchor, proposals)
        full_ids = full_logits.argmax(-1).cpu()
        mismatches = proposals.cpu().ne(full_ids)
        event = dict(request=current_request, prefix_length=features.shape[1],
                     proposals=proposals.cpu().tolist()[0],
                     training_forward_tokens=full_ids.tolist()[0],
                     differing_positions=mismatches[0].nonzero().flatten().tolist(),
                     cached_vs_training_logits=difference(cached_logits, full_logits))
        if remote is not None:
            # LocalTarget.tokens may still include an unaccepted verification
            # suffix. The feature length identifies the verified prefix exactly.
            prefix_tokens = target.tokens[:features.shape[1]]
            remote_states = remote.states(prefix_tokens)
            remote_ids = original(remote_states.auxiliary, anchor, proposals.shape[1])
            # Compare draft scores under an identical token history, even if
            # the two freely generated proposal blocks diverge.
            remote_logits = replay_training_forward(draft, remote_states.auxiliary, anchor, proposals)
            hf_scoring = target.last.scoring[:, :features.shape[1]]
            hf_top = draft.verifier_lm_head(hf_scoring[:, -1:]).argmax(-1)
            remote_top = draft.verifier_lm_head(remote_states.scoring[:, -1:]).argmax(-1)
            event['hf_vs_vllm'] = dict(
                target_features=difference(features, remote_states.auxiliary),
                target_last_scoring=difference(hf_scoring[:, -1:], remote_states.scoring[:, -1:]),
                target_next_token_equal=bool(torch.equal(hf_top, remote_top)),
                remote_proposals=remote_ids.cpu().tolist()[0],
                free_running_proposals_equal=bool(torch.equal(proposals, remote_ids)),
                same_history_draft_logits=difference(full_logits, remote_logits),
                same_history_draft_argmax_mismatches=int(full_ids.ne(remote_logits.argmax(-1).cpu()).sum()),
            )
        events.append(event)
        (output / 'rounds.json').write_text(json.dumps(events, indent=2) + '\n')
        print(f"Audit round {len(events)}: prefix={features.shape[1]}, "
              f"cached/training mismatches={len(event['differing_positions'])}", flush=True)
        return proposals

    with open(a.prompts) as stream, patch.object(draft, 'propose', side_effect=audited):
        for index, line in enumerate(stream):
            if index >= a.num_prompts or len(events) >= a.rounds:
                break
            current_request = index
            row = json.loads(line)
            ids = row.get('input_ids')
            if ids is None:
                ids = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
            result, _ = generate(target, draft, ids, a.max_new_tokens, eos)
            reference = target_greedy(target, draft, ids, a.max_new_tokens, eos)
            requests.append(dict(index=index, greedy_output_equal=result == reference))
    source_paths = [Path(__file__), Path(__file__).parent.parent / 'src/speculators/models/dflash_ar/core.py',
                    Path(__file__).parent.parent / 'src/speculators/models/retrace/target.py']
    summary = dict(arguments=vars(a), recorded_target=recorded_target, actual_target=target_path,
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        checkpoint_keys=keys, source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in source_paths},
        checked_rounds=len(events),
        cached_training_token_mismatches=sum(len(e['differing_positions']) for e in events),
        nonfinite_rounds=sum(not e['cached_vs_training_logits']['finite'] for e in events),
        requests=requests,
        note='Diagnostic only. CPU parity does not establish NPU parity. Small numeric drift is not by itself a bug.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not events or summary['nonfinite_rounds'] or summary['cached_training_token_mismatches'] or any(
            not request['greedy_output_equal'] for request in requests):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
