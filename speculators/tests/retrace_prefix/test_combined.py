"""CPU tests: no downloads, target server, accelerator or external checkpoints."""
from concurrent.futures import Future

import pytest
import torch
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash_prefix import DFlashPrefixDraftModel, DFlashPrefixSpeculatorConfig
from speculators.models.retrace.memory import ReTraceMemory
from speculators.models.retrace_prefix import ReTracePrefixDraftModel, ReTracePrefixSpeculatorConfig
from speculators.models.retrace_prefix.loading import load_config
from speculators.models.retrace_prefix.loss import topk_kl
from speculators.models.retrace_prefix.stored_training import ReTracePrefixTraining
from speculators.proposals.greedy import GreedyTokenProposalConfig


def config(kind="compiled_prefix_v3", target=None, combined=True):
    layer = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=16, layer_types=["full_attention"], _attn_implementation="sdpa")
    cls = ReTracePrefixSpeculatorConfig if combined else DFlashPrefixSpeculatorConfig
    return cls(transformer_layer_config=layer, draft_vocab_size=32, block_size=4,
        aux_hidden_state_layer_ids=[1, 2], mask_token_id=0, sample_from_anchor=False,
        prefix_selector_kind=kind, prefix_rank=16, prefix_top_k=16,
        prefix_attention_heads=2, prefix_attention_layers=1,
        speculators_config=SpeculatorsConfig(algorithm="retrace_prefix" if combined else "dflash_prefix",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy", verifier=VerifierConfig(
                name_or_path=str(target) if target else None, architectures=["Qwen3ForCausalLM"])))


def test_topk_kl_math_gradient_mask():
    torch.manual_seed(2)
    student = torch.randn(2, 3, 32, requires_grad=True)
    teacher = torch.randn(2, 3, 32, requires_grad=True)
    valid = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    loss = topk_kl(student, teacher, valid)
    ids = teacher.detach().topk(16, dim=-1).indices
    p = teacher.detach().gather(-1, ids).softmax(-1)
    q = student.gather(-1, ids).softmax(-1)
    expected = ((p * (p.log()-q.log())).sum(-1) * valid * torch.exp(-torch.arange(3)/4)).sum()/valid.sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[~valid].count_nonzero() == 0
    outside = torch.ones_like(student, dtype=torch.bool).scatter(-1, ids, False)
    assert student.grad[outside].count_nonzero() == 0
    assert student.grad.count_nonzero() > 0
    torch.testing.assert_close(topk_kl(teacher, teacher, valid), torch.tensor(0.))
    assert topk_kl(student, teacher, torch.zeros_like(valid)).item() == 0


@pytest.mark.parametrize("kind", ["legacy", "local_prefix_v2", "compiled_prefix_v3"])
def test_block_joint_backward_and_roundtrip(tmp_path, kind):
    torch.manual_seed(4)
    target_dir = make_target(tmp_path)
    model = ReTracePrefixTraining(config(kind, target_dir))
    model.load_verifier_weights()
    context = torch.randn(1, 9, 64)
    anchors, positions = torch.tensor([1, 2]), torch.tensor([3, 6])
    memory = ReTraceMemory(torch.randn(2, 3, 32), torch.randn(2, 3, 32), torch.ones(2, 3, dtype=torch.bool))
    block = model.training_blocks(context, anchors, positions, memory)
    assert block.token_ids.shape == (2, 3)
    valid = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    target = torch.randn(2, 3, 32)
    loss, metrics = model.auxiliary_loss(block, target, valid, anchors, torch.ones(2,3,dtype=torch.long))
    torch.testing.assert_close(loss, model.config.prefix_loss_alpha*metrics['selector_ce'] + .1*metrics['top16_kl'])
    loss.backward()
    for prefix in ('layers.', 'prefix_head.', 'retrace.'):
        grads = [p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.count_nonzero() for g in grads)
    # Same outputs with missing memory and with zero-initialized residual fusion.
    plain = model.training_blocks(context, anchors, positions, None)
    torch.testing.assert_close(plain.logits, block.logits)
    model.save_pretrained(tmp_path)
    saved = load_config(tmp_path)
    assert saved.distill_top_k == 16 and saved.distill_kl_weight == .1
    restored = ReTracePrefixDraftModel.from_pretrained(tmp_path, config=saved)
    again = restored.training_blocks(context, anchors, positions, memory)
    torch.testing.assert_close(again.logits, block.logits)
    torch.testing.assert_close(again.token_ids, block.token_ids)


def test_stored_forward_verifies_selected_path_and_trains_conditioner(tmp_path):
    torch.manual_seed(8)
    model = ReTracePrefixTraining(config(target=make_target(tmp_path)))
    model.load_verifier_weights()
    # Force a deterministic source proposal that rejects immediately; correction
    # matches the stored response. This exercises real source -> successor flow.
    selected = torch.ones(1, 3, dtype=torch.long)
    model.prefix_head.greedy = lambda hidden, *args, **kw: selected.expand(hidden.shape[0], -1)
    model.verifier_lm_head = torch.nn.Linear(32, 32, bias=False).requires_grad_(False)
    model.verifier_lm_head.weight.data.zero_()  # target argmax = 0
    class Target:
        def __init__(self): self.requests = []
        def submit(self, tokens, anchor, k):
            self.requests.append((tokens, anchor, k))
            f = Future(); f.set_result(torch.randn(k, 32)); return f
    teacher = Target()
    model.configure_stored(teacher, pairs_per_batch=2, blocks_per_forward=2)
    model.step_provider = lambda: 100
    _, loss, metrics = model(
        input_ids=torch.zeros(1, 16, dtype=torch.long),
        hidden_states=torch.randn(1, 16, 64),
        verifier_last_hidden_states=torch.randn(1, 16, 32),
        loss_mask=torch.ones(1, 16), document_ids=torch.zeros(1, 16,dtype=torch.long))
    assert teacher.requests and all(tokens[-3:] == [1,1,1] for tokens,_,_ in teacher.requests)
    assert metrics['conditioned_pair_fraction_sum'] == 2
    assert metrics['top16_kl_total'] == 1
    loss.backward()
    assert model.retrace.value.weight.grad is not None
    assert torch.isfinite(loss)


def make_target(tmp_path):
    from safetensors.torch import save_file
    target = tmp_path / "target"
    target.mkdir()
    cfg = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        head_dim=16, architectures=["Qwen3ForCausalLM"])
    cfg.save_pretrained(target)
    save_file({"model.embed_tokens.weight": torch.randn(32,32)*.2,
               "lm_head.weight": torch.randn(32,32)*.2,
               "model.norm.weight": torch.ones(32)}, str(target / "model.safetensors"))
    return target


def test_import_preserves_head_and_backbone(tmp_path):
    from speculators.models.retrace.pretrained import load_dflash, prepare
    target = make_target(tmp_path)
    source = DFlashPrefixDraftModel(config(target=target, combined=False))
    source.load_verifier_weights()
    source.save_pretrained(tmp_path / "source")
    combined, report = load_dflash(tmp_path / "source", target, dtype=torch.float32)
    assert combined.config.speculators_model_type == "retrace_prefix"
    assert report["residual_projection_is_zero"]
    for name, value in source.state_dict().items():
        torch.testing.assert_close(combined.state_dict()[name], value, rtol=0, atol=0)
    prepare(tmp_path / "source", target, tmp_path / "converted")
    cfg = load_config(tmp_path / "converted/retrace")
    assert cfg.speculators_model_type == "retrace_prefix"
    baseline = DFlashPrefixSpeculatorConfig.from_pretrained(tmp_path / "converted/dflash")
    assert baseline.speculators_model_type == "dflash_prefix"


def test_bfloat16_autocast_retains_selector_gradients(tmp_path):
    model = ReTracePrefixTraining(config(target=make_target(tmp_path)))
    model.load_verifier_weights()
    for p in model.parameters():
        if not p.requires_grad:
            p.data = p.data.bfloat16()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        block = model.training_blocks(torch.randn(1, 8, 64), torch.tensor([1]), torch.tensor([3]), None)
        loss, _ = model.auxiliary_loss(block, torch.randn(1,3,32), torch.ones(1,3,dtype=torch.bool), torch.tensor([1]), torch.ones(1,3,dtype=torch.long))
    loss.backward()
    for name, p in model.named_parameters():
        if name.startswith('prefix_head.') and p.requires_grad:
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name


def test_no_future_context_leakage(tmp_path):
    model = ReTracePrefixDraftModel(config(target=make_target(tmp_path)))
    model.load_verifier_weights()
    context = torch.randn(1, 9, 64)
    original = model.training_blocks(context, torch.tensor([1]), torch.tensor([3]), None)
    context[:, 3:] = torch.randn_like(context[:, 3:])*100
    changed = model.training_blocks(context, torch.tensor([1]), torch.tensor([3]), None)
    torch.testing.assert_close(original.logits, changed.logits, rtol=0, atol=0)


def test_auxiliary_chunk_invariance(tmp_path):
    from speculators.models.retrace.core import DraftBlock
    model = ReTracePrefixTraining(config(target=make_target(tmp_path)))
    model.load_verifier_weights()
    block = model.training_blocks(torch.randn(1, 12, 64), torch.tensor([1,2,3]), torch.tensor([2,5,8]), None)
    targets = torch.randn(3,3,32)
    valid = torch.tensor([[1,1,1], [1,1,0], [1,0,0]],dtype=torch.bool)
    anchors, reference = torch.tensor([1,2,3]), torch.zeros(3,3,dtype=torch.long)
    full, _ = model.auxiliary_loss(block, targets, valid, anchors, reference)
    chunks = 0.
    for i in range(3):
        part = DraftBlock(block.hidden[i:i+1], block.logits[i:i+1], block.token_ids[i:i+1])
        loss, _ = model.auxiliary_loss(part, targets[i:i+1], valid[i:i+1], anchors[i:i+1], reference[i:i+1])
        chunks = chunks + loss * valid[i].sum()/valid.sum()
    torch.testing.assert_close(full, chunks)


def test_vanilla_retrace_import_unchanged(tmp_path):
    from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
    from speculators.models.retrace.pretrained import load_dflash
    target = make_target(tmp_path)
    raw = config(target=target, combined=False).to_dict()
    raw = {k:v for k,v in raw.items() if not k.startswith('prefix_')}
    raw.update(speculators_model_type='dflash', architectures=['DFlashDraftModel'])
    raw['speculators_config']['algorithm'] = 'dflash'
    source = DFlashDraftModel(DFlashSpeculatorConfig(**raw))
    source.load_verifier_weights()
    source.save_pretrained(tmp_path/'source')
    result, _ = load_dflash(tmp_path/'source', str(target), dtype=torch.float32)
    assert result.config.speculators_model_type == 'retrace'
    assert not hasattr(result, 'prefix_head')
    for name, value in source.state_dict().items():
        torch.testing.assert_close(result.state_dict()[name], value, rtol=0, atol=0)
