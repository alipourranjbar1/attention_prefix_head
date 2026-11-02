from types import SimpleNamespace

from four_anchor.vllm_entry import engine_options, server_command, summarize


def test_serving_config_preserves_four_anchors_and_uses_native_worker(components, tmp_path):
    _target, draft, _runner = components
    draft.save(tmp_path)
    args = SimpleNamespace(target="/local/qwen", max_model_len=4096, memory_fraction=.8,
                           checkpoint=str(tmp_path), proposals=15, mode="two_pass")
    options = engine_options(args)
    assert options["speculative_config"]["num_speculative_tokens"] == 15
    assert options["hf_overrides"]["four_anchor_serving"]["num_proposals"] == 15
    assert "additional_config" not in options  # Ascend forbids foreign keys here.
    command = server_command(options, "127.0.0.1", 8095)
    for flag in ("--worker-cls", "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
                 "--no-async-scheduling", "--enforce-eager", "--hf-overrides"):
        assert flag in command
    assert options["worker_cls"] == "four_anchor.vllm_ascend.FourAnchorWorker"


def test_ar_engine_has_no_checkpoint_or_speculator():
    args = SimpleNamespace(target="/local/qwen", max_model_len=4096, memory_fraction=.8, mode="ar")
    options = engine_options(args)
    assert not any(k in options for k in ("speculative_config", "worker_cls", "hf_overrides"))


def test_metrics_pool_tokens_time_and_verified_cycles():
    rows = [{"output_tokens": 100, "seconds": 4,
             "speculative_decoding": {"num_spec_steps": 10, "num_accepted_draft_tokens": 90}},
            {"output_tokens": 20, "seconds": 2,
             "speculative_decoding": {"num_spec_steps": 5, "num_accepted_draft_tokens": 15}}]
    summary = summarize(rows)
    assert summary["tokens_per_second_including_prefill"] == 20
    assert summary["accepted_draft_per_verification"] == 7
    assert summary["acceptance_length_including_bonus"] == 8
