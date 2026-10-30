#!/usr/bin/env python3
"""Check local runtime source behavior without loading vLLM/NPU dependencies.

--npu-check additionally compares the actual paged Ascend FIA kernel against
an explicit full-context, causal-block reference. It requires a free NPU.
"""

import argparse
import ast
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


def source_functions(relative, names, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    if {node.name for node in selected} != set(names):
        raise RuntimeError(f"Required runtime functions missing: {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def runtime_functions():
    namespace = {"Qwen3Config": NS, "_SLIDING_ATTENTION": "sliding_attention"}
    source_functions("vllm/vllm/transformers_utils/configs/speculators/algos.py", ["update_dflash"], namespace)
    source_functions("vllm/vllm/model_executor/models/qwen3_dflash.py",
                     ["_dflash_layer_causal", "dflash_has_any_non_causal"], namespace)
    return namespace


def converted_flags(config):
    functions = runtime_functions()
    converted = copy.deepcopy(config["transformer_layer_config"])
    functions["update_dflash"](config, converted)
    hf = NS(**converted)
    return [functions["_dflash_layer_causal"](hf, i) for i in range(hf.num_hidden_layers)]


def sample_config():
    return {
        "speculators_model_type": "dflash", "last_layer_causal": True,
        "sliding_window_non_causal": False, "mask_token_id": 151669,
        "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33],
        "transformer_layer_config": {"num_hidden_layers": 5, "layer_types": ["full_attention"] * 5},
    }


class FakeBase:
    """Only replace the NPU-dependent superclass, not the new routing code."""
    shallow_copy_metadata = staticmethod(copy.copy)

    def build_draft_attn_metadata(self, *_):
        self.shared = NS(causal=False, attn_mask=None, slot_mapping=object())
        return [{name: self.shared for name in self.names}], self.shared


def proposer(flags=None, eager=True, names=None):
    path = ROOT / "vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendDflashProposer")
    required = {"_dflash_attention_is_causal", "build_draft_attn_metadata"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in required]
    if {node.name for node in cls.body} != required:
        raise RuntimeError("Apply the Ascend last-causal patch first")
    namespace = {"AscendEagleProposer": FakeBase}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    obj = namespace["AscendDflashProposer"]()
    flags = [False] * 4 + [True] if flags is None else flags
    obj.names = names or [f"draft.layers.{i}.attn" for i in range(len(flags))]
    obj.model = NS(get_draft_attn_causal=lambda: flags, get_draft_kv_cache_layer_names=lambda: obj.names)
    obj.speculative_config = NS(enforce_eager=eager)
    obj.vllm_config = NS(parallel_config=NS(prefill_context_parallel_size=1, decode_context_parallel_size=1))
    obj.causal_mask = object()
    builder = NS(attn_mask_builder=NS(get_attention_mask=lambda causal, config: obj.causal_mask if causal else None),
                 model_config=NS())
    obj.draft_attn_groups = [NS(layer_names=list(reversed(obj.names)), get_metadata_builder=lambda: builder)]
    return obj


class RuntimeTests(unittest.TestCase):
    def test_checkpoint_sliding_flag_does_not_make_every_layer_causal(self):
        for sliding_flag in (False, True):
            config = sample_config()
            config["sliding_window_non_causal"] = sliding_flag
            self.assertEqual(converted_flags(config), [False] * 4 + [True])

    def test_legacy_override_is_preserved_without_new_flag(self):
        config = sample_config()
        config["last_layer_causal"] = False
        self.assertEqual(converted_flags(config), [True] * 5)

    def test_final_layer_has_its_own_metadata_and_mask(self):
        obj = proposer(names=[f"draft.layers.{i}.attn" for i in (0, 1, 2, 3, 10)])
        metadata, _ = obj.build_draft_attn_metadata(None, 16, 16)
        self.assertFalse(obj._dflash_attention_is_causal())
        for name in obj.names[:-1]:
            self.assertIs(metadata[0][name], obj.shared)
            self.assertFalse(metadata[0][name].causal)
            self.assertIsNone(metadata[0][name].attn_mask)
        last = metadata[0][obj.names[-1]]
        self.assertIsNot(last, obj.shared)
        self.assertTrue(last.causal)
        self.assertIs(last.attn_mask, obj.causal_mask)
        self.assertIs(last.slot_mapping, obj.shared.slot_mapping)

    def test_uniform_layers_keep_existing_metadata(self):
        obj = proposer(flags=[False] * 5)
        metadata, _ = obj.build_draft_attn_metadata(None, 16, 16)
        self.assertTrue(all(value is obj.shared for value in metadata[0].values()))

    def test_draft_graph_and_context_parallel_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "enforce_eager"):
            proposer(eager=False)._dflash_attention_is_causal()
        obj = proposer()
        obj.vllm_config.parallel_config.decode_context_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "context parallel"):
            obj._dflash_attention_is_causal()

    def test_unsupported_architecture_fails_clearly(self):
        cfg = sample_config()
        cfg["speculators_model_type"] = "dflash_prefix"
        with self.assertRaisesRegex(ValueError, "plain DFlash"):
            converted_flags(cfg)
        cfg = sample_config()
        cfg["transformer_layer_config"]["layer_types"][0] = "sliding_attention"
        with self.assertRaisesRegex(ValueError, "full-context"):
            converted_flags(cfg)


def validate_checkpoint(checkpoint):
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    expected = [False] * 4 + [True]
    if config.get("last_layer_causal") is not True or converted_flags(config) != expected:
        raise ValueError("Expected a saved five-layer plain DFlash checkpoint with last_layer_causal=true")
    if config.get("block_size") != 16 or config.get("sample_from_anchor", False):
        raise ValueError("This evaluation recipe requires block_size=16, sample_from_anchor=false")
    weights = list(checkpoint.glob("*.safetensors"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise ValueError("Checkpoint weights are missing or empty; wait for saving to finish")
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        for filename in set(json.loads(index.read_text())["weight_map"].values()):
            if not (checkpoint / filename).is_file():
                raise ValueError(f"Checkpoint shard missing: {filename}")
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Serving layer masks: {expected} (False=bidirectional, True=causal)", flush=True)


def npu_check():
    import torch
    import torch_npu
    from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions

    _register_atb_extensions()
    device = "npu:0"
    torch.manual_seed(42)
    block, heads, kv_heads, dim, cache_block = 16, 2, 1, 128, 128
    sparse_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).to(device)
    for context in (17, 273):
        length = context + block
        pages = (length + cache_block - 1) // cache_block
        q = torch.randn(block, heads, dim, dtype=torch.bfloat16).to(device)
        k = torch.randn(pages, cache_block, kv_heads * dim, dtype=torch.bfloat16).to(device)
        v = torch.randn_like(k)
        table = torch.arange(pages, dtype=torch.int32, device=device).unsqueeze(0)
        q_cpu = q.cpu().float().transpose(0, 1)
        k_cpu = k.cpu().float().reshape(-1, kv_heads, dim)[:length].transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
        v_cpu = v.cpu().float().reshape(-1, kv_heads, dim)[:length].transpose(0, 1).repeat_interleave(heads // kv_heads, 0)
        scores = q_cpu @ k_cpu.transpose(-2, -1) / dim**0.5
        allowed = torch.arange(length)[None, :] <= context + torch.arange(block)[:, None]
        for causal in (False, True):
            kwargs = dict(atten_mask=sparse_mask, sparse_mode=3) if causal else dict(sparse_mode=0)
            actual, _ = torch_npu.npu_fused_infer_attention_score(
                query=q, key=k, value=v, block_table=table, input_layout="TND",
                block_size=cache_block, actual_seq_lengths=[block], actual_seq_lengths_kv=[length],
                num_heads=heads, num_key_value_heads=kv_heads, scale=dim**-0.5, **kwargs,
            )
            logits = scores.masked_fill(~allowed, float("-inf")) if causal else scores
            expected = (logits.softmax(-1) @ v_cpu).transpose(0, 1)
            torch.testing.assert_close(actual.cpu().float().reshape_as(expected), expected, atol=0.03, rtol=0.03)
    print("NPU paged attention matches the full-context + causal-block reference: OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--npu-check", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.checkpoint:
        validate_checkpoint(args.checkpoint)
    if args.npu_check:
        npu_check()


if __name__ == "__main__":
    main()
