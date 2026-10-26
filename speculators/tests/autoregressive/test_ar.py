import copy
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dflash_ar import DFlashARDraftModel, DFlashARSpeculatorConfig
from speculators.models.dflash_ar.evaluate import generate, target_greedy
from speculators.models.retrace.target import LocalTarget
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.config import TrainConfig


def models():
    torch.manual_seed(7)
    target_cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=3, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8,
                            layer_types=['full_attention'] * 3,
                            _attn_implementation='eager')
    target = Qwen3ForCausalLM(target_cfg).eval()
    decoder = copy.deepcopy(target_cfg)
    decoder.num_hidden_layers = 2
    decoder.layer_types = ['full_attention'] * 2
    cfg = DFlashARSpeculatorConfig(
        transformer_layer_config=decoder, draft_vocab_size=32, block_size=4,
        aux_hidden_state_layer_ids=[1, 2], mask_token_id=0,
        speculators_config=SpeculatorsConfig(algorithm='dflash_ar',
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=4)],
            default_proposal_method='greedy',
            verifier=VerifierConfig(name_or_path=None, architectures=['Qwen3ForCausalLM'])))
    draft = DFlashARDraftModel(cfg).eval()
    draft.embed_tokens = target.model.embed_tokens
    draft.lm_head = target.lm_head
    draft.verifier_lm_head = target.lm_head
    draft.verifier_norm = target.model.norm
    return target, draft


class ARTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_layout_identical_to_dflash(self):
        _, ar = models()
        data = ar.config.model_dump()
        data.update(speculators_model_type='dflash', architectures=['DFlashSpeculator'], sample_from_anchor=False)
        data['speculators_config']['algorithm'] = 'dflash'
        base = DFlashDraftModel(DFlashSpeculatorConfig(**data))
        self.assertEqual({k: tuple(v.shape) for k, v in base.state_dict().items()},
                         {k: tuple(v.shape) for k, v in ar.state_dict().items()})

    def batch(self, ar):
        length = 12
        return dict(hidden_states=torch.randn(1, length, 32),
                    input_ids=torch.randint(0, 32, (1, length)),
                    loss_mask=torch.ones(1, length),
                    verifier_last_hidden_states=torch.randn(1, length, 16),
                    document_ids=torch.zeros(1, length, dtype=torch.long), max_anchors=1)

    def fixed(self, ar, batch, anchor=3):
        pos = torch.tensor([anchor])
        mask = ar._create_attention_mask(batch['document_ids'], 12, pos, torch.device('cpu'))
        return patch.object(ar, '_build_attention_mask', return_value=(mask, None, pos, torch.tensor([True])))

    def test_future_tokens_and_target_features_cannot_leak(self):
        _, ar = models()
        b = self.batch(ar)
        with self.fixed(ar, b):
            first = ar._backbone_forward(**b)[1]
            changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            changed['input_ids'][:, 5:] = (changed['input_ids'][:, 5:] + 5) % 32
            changed['hidden_states'][:, 3:] += 100
            second = ar._backbone_forward(**changed)[1]
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_cached_rollout_matches_teacher_forcing_on_its_own_tokens(self):
        _, ar = models()
        b = self.batch(ar)
        proposals = ar.propose(b['hidden_states'][:, :3], b['input_ids'][:, 3:4])
        b['input_ids'][:, 4:8] = proposals
        with self.fixed(ar, b):
            logits = ar._backbone_forward(**b)[1]
        self.assertEqual(logits.argmax(-1).tolist(), proposals.tolist())

    def test_sdpa_and_sliding_cache_match_training(self):
        for sliding in (False, True):
            _, ar = models()
            ar._attn_impl = 'sdpa'
            ar.config.transformer_layer_config._attn_implementation = 'sdpa'
            from torch.nn.attention.flex_attention import create_mask
            ar._create_mask_fn = create_mask
            if sliding:
                ar.sliding_window = 2
                ar.sliding_window_indices = [0, 1]
                ar.uses_sliding_window_attn = True
                ar.uses_full_attn = False
            b = self.batch(ar)
            anchor = torch.tensor([3])
            mask = ar._create_attention_mask(b['document_ids'], 12, anchor,
                                             torch.device('cpu'), 2 if sliding else None)
            proposals = ar.propose(b['hidden_states'][:, :3], b['input_ids'][:, 3:4])
            b['input_ids'][:, 4:8] = proposals
            masks = (None, mask) if sliding else (mask, None)
            with patch.object(ar, '_build_attention_mask', return_value=(*masks, anchor, torch.tensor([True]))):
                logits = ar._backbone_forward(**b)[1]
            self.assertEqual(logits.argmax(-1).tolist(), proposals.tolist())

    def test_next_token_labels_and_document_boundary(self):
        _, ar = models()
        b = self.batch(ar)
        b['document_ids'][:, 6:] = 1
        b['loss_mask'][:, 4] = 0
        with self.fixed(ar, b):
            _, _, labels, mask, _ = ar._backbone_forward(**b)
        expected = ar.verifier_lm_head(ar.verifier_norm(b['verifier_last_hidden_states'][:, 3:7]))
        torch.testing.assert_close(labels, expected)
        self.assertEqual(mask.tolist(), [[0., 1., 0., 0.]])

    def test_gradients_and_teacher_metric_name(self):
        _, ar = models()
        ar.train()
        _, loss, metrics = ar(**self.batch(ar), per_position_loss_weight='fixed-exp-decay', loss_config=resolve_loss_config('ce', 'eager'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(ar.fc.weight.grad.abs().sum().item(), 0)
        self.assertIn('teacher_forced_eal_sum', metrics)
        self.assertNotIn('eal_sum', metrics)

    def test_free_running_matches_target_and_eos(self):
        target, ar = models()
        engine = LocalTarget(target, ar.target_layer_ids)
        prompt = [1, 2, 3]
        ref = target_greedy(engine, ar, prompt, 20)
        out, rounds = generate(engine, ar, prompt, 20)
        self.assertEqual(out, ref)
        self.assertTrue(rounds)
        out, _ = generate(engine, ar, prompt, 20, {ref[0]})
        self.assertEqual(out, ref[:1])

    def test_all_accept_and_mismatch_at_each_position(self):
        target, ar = models()
        engine = LocalTarget(target, ar.target_layer_ids)
        prompt = [1, 2, 3]
        ref = target_greedy(engine, ar, prompt, 20)
        for mismatch in range(5):
            def proposal(features, anchor, count, mismatch=mismatch):
                offset = features.shape[1] - len(prompt) + 1
                ids = ref[offset:offset + count].copy()
                if mismatch < count:
                    ids[mismatch] = (ids[mismatch] + 1) % 32
                return torch.tensor([ids])
            with patch.object(ar, 'propose', side_effect=proposal):
                out, rounds = generate(engine, ar, prompt, 20)
            self.assertEqual(out, ref)
            self.assertEqual(rounds[0]['accepted'], mismatch)

    def test_checkpoint_roundtrip_reloads_target_owned_weights(self):
        target, ar = models()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target.save_pretrained(root / 'target')
            ar.config.speculators_config.verifier.name_or_path = str(root / 'target')
            ar.save_pretrained(root / 'draft')
            cfg = DFlashARSpeculatorConfig.from_pretrained(root / 'draft')
            cfg.transformer_layer_config._attn_implementation = 'eager'
            loaded = DFlashARDraftModel.from_pretrained(root / 'draft', config=cfg,
                                                       verifier=str(root / 'target')).eval()
            features = torch.randn(1, 3, 32)
            anchor = torch.tensor([[4]])
            self.assertEqual(ar.propose(features, anchor).tolist(), loaded.propose(features, anchor).tolist())

    def test_recipe_preparation_preserves_training_settings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            baseline = TrainConfig.resolve(['--speculator-type', 'dflash',
                '--verifier-name-or-path', '/unused', '--data-path', '/data',
                '--epochs', '5', '--lr', '0.0006', '--total-seq-len', '3072',
                '--num-layers', '5', '--target-layer-ids', '1', '9', '17', '25', '33'])
            source = root / 'baseline.yaml'
            source.write_text(baseline.dump_yaml())
            script = Path(__file__).resolve().parents[2] / 'scripts' / 'prepare_dflash_ar.py'
            subprocess.run([sys.executable, str(script), '--baseline-config', str(source),
                            '--output', str(root / 'out')], check=True, capture_output=True)
            ar = TrainConfig.resolve(['--config', str(root / 'out' / 'train_config.yaml')])
            before, after = baseline.flatten(), ar.flatten()
            for key in ('lr', 'epochs', 'optimizer', 'total_seq_len', 'target_layer_ids',
                        'num_layers', 'max_anchors', 'loss_fn', 'loss_implementation',
                        'per_position_loss_weight', 'train_data_ratio'):
                self.assertEqual(before[key], after[key], key)
            self.assertTrue(after['sample_from_anchor'])
            self.assertEqual(after['speculator_type'], 'dflash_ar')

    def test_config_registry_and_defaults(self):
        cfg = TrainConfig.resolve(['--speculator-type', 'dflash_ar', '--verifier-name-or-path', '/unused'])
        self.assertEqual(cfg.flatten()['num_layers'], 5)
        self.assertEqual(cfg.flatten()['block_size'], 16)
        _, ar = models()
        with tempfile.TemporaryDirectory() as d:
            ar.config.save_pretrained(d)
            cfg2 = DFlashARSpeculatorConfig.from_pretrained(d)
            self.assertEqual(cfg2.speculators_model_type, 'dflash_ar')


if __name__ == '__main__':
    unittest.main()
