"""Acceptance measurement, not a vLLM throughput benchmark. JSONL prompts."""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from speculators.models.dflash_ar import DFlashARDraftModel, DFlashARSpeculatorConfig
from speculators.models.dflash_ar.evaluate import generate, target_greedy
from speculators.models.retrace.target import LocalTarget
from speculators.train.utils import save_train_command


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--target', required=True)
    p.add_argument('--draft', required=True)
    p.add_argument('--prompts', required=True, help='JSONL: {"prompt": "..."} or {"input_ids": [...]}')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--num-prompts', type=int, default=128)
    p.add_argument('--check-greedy', type=int, default=3)
    a = p.parse_args()
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        torch.npu.set_device(a.device)
    dtype = torch.float32 if a.device == 'cpu' else torch.bfloat16
    cfg = DFlashARSpeculatorConfig.from_pretrained(a.draft)
    cfg.transformer_layer_config._attn_implementation = 'sdpa'
    draft = DFlashARDraftModel.from_pretrained(a.draft, config=cfg, verifier=a.target,
                                              torch_dtype=dtype).to(a.device).eval()
    model = AutoModelForCausalLM.from_pretrained(a.target, torch_dtype=dtype,
                                               attn_implementation='sdpa').to(a.device).eval()
    # Same target-owned parameters; also avoid retaining redundant NPU copies.
    draft.embed_tokens = model.model.embed_tokens
    draft.lm_head = model.lm_head
    draft.verifier_lm_head = model.lm_head
    draft.verifier_norm = model.model.norm
    if max(draft.target_layer_ids) >= model.config.num_hidden_layers:
        raise ValueError('Reference HF adapter requires auxiliary layers before final norm')
    target = LocalTarget(model, draft.target_layer_ids)
    tokenizer = AutoTokenizer.from_pretrained(a.target)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    # Reuse repo provenance writer; includes argv, versions, SHA and local patch.
    save_train_command(out)
    rows, all_rounds = [], []
    with open(a.prompts) as f, (out / 'requests.jsonl').open('w') as log:
        for index, line in enumerate(f):
            if index >= a.num_prompts:
                break
            item = json.loads(line)
            ids = item.get('input_ids')
            if ids is None:
                ids = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': item['prompt']}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
            result, rounds = generate(target, draft, ids, a.max_new_tokens, eos)
            parity = None
            if index < a.check_greedy:
                parity = result == target_greedy(target, draft, ids, a.max_new_tokens, eos)
                if not parity:
                    raise RuntimeError(f'Target-only greedy parity failed on prompt {index}; do not use results')
            row = dict(index=index, output_ids=result, text=tokenizer.decode(result),
                       greedy_parity=parity, rounds=rounds)
            log.write(json.dumps(row) + '\n')
            log.flush()
            rows.append(row)
            all_rounds.extend(rounds)
            print(f'Completed {index + 1}: {len(result)} output tokens, {len(rounds)} verification rounds', flush=True)
    full = [r for r in all_rounds if r['full_block']]
    def stats(rs):
        if not rs:
            return dict(rounds=0, acceptance_length_with_bonus=None, draft_acceptance_rate=None)
        return dict(rounds=len(rs),
                    acceptance_length_with_bonus=1 + sum(r['accepted'] for r in rs) / len(rs),
                    draft_acceptance_rate=sum(r['accepted'] for r in rs) / sum(r['proposed'] for r in rs),
                    accepted_histogram={str(i): sum(r['accepted'] == i for r in rs)
                                        for i in range(draft.block_size + 1)})
    summary = dict(requests=len(rows), speculative_tokens=draft.block_size,
                   full_block=stats(full), all_rounds=stats(all_rounds),
                   note='Greedy free-running acceptance; raw rounds include EOS suffixes. Full-block statistics exclude token-budget-shortened blocks. Reference implementation is not a vLLM speed benchmark.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
