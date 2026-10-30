import random

import pytest
from datasets import Dataset
from torch.utils.data.distributed import DistributedSampler

from four_anchor.data import filter_training_records, training_example


def test_filter_matches_truncated_prediction_window(tmp_path):
    data = Dataset.from_list([
        {"input_ids": [], "loss_mask": []},
        {"input_ids": [1], "loss_mask": [1.]},
        {"input_ids": [1, 2, 3], "loss_mask": [0., 0., 0.]},
        {"input_ids": [1, 2, 3], "loss_mask": [1., 0., 0.]},
        {"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0., 0., 0., 0., 1.]},
        {"input_ids": [1, 2, 3, 4], "loss_mask": [0., 0., 0., 0.25]},
        {"input_ids": [6, 7], "loss_mask": [0., 1.]},
    ])
    filtered, report = filter_training_records(data, 4, tmp_path)
    assert (report["input_rows"], report["trainable_rows"], report["dropped_rows"]) == (7, 2, 5)
    assert [row["input_ids"] for row in filtered] == [[1, 2, 3, 4], [6, 7]]
    for index in range(len(filtered)):
        prefix, labels, mask = training_example(filtered, random.Random(42), 16, 4, 64, "cpu", row_index=index)
        assert prefix.shape[1] > 0 and mask.sum() > 0
    assert len(data) == 7  # source remains unchanged


def test_cache_reuse_and_fingerprint_independent_of_output_path(tmp_path, monkeypatch):
    import four_anchor.data as module
    data = Dataset.from_list([{"input_ids": [1, 2], "loss_mask": [0, 1]},
                              {"input_ids": [1, 2], "loss_mask": [0, 0]}])
    a, _ = filter_training_records(data, 4, tmp_path / "a")
    b, _ = filter_training_records(data, 4, tmp_path / "b")
    assert a._fingerprint == b._fingerprint
    def unexpected_scan(*args, **kwargs):
        raise AssertionError("The second rank should load cached indices without rescanning")
    monkeypatch.setattr(module, "_trainable_batch", unexpected_scan)
    cached, _ = filter_training_records(data, 4, tmp_path / "a")
    assert cached._fingerprint == a._fingerprint
    assert cached[0] == a[0]


def test_cache_separates_truncation_lengths(tmp_path):
    data = Dataset.from_list([{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 0, 1]},
                              {"input_ids": [5, 6], "loss_mask": [0, 1]}])
    short, _ = filter_training_records(data, 3, tmp_path)
    long, _ = filter_training_records(data, 4, tmp_path)
    assert len(short) == 1 and len(long) == 2
    assert long._fingerprint == data._fingerprint


def test_bool_mask_and_unmasked_records(tmp_path):
    data = Dataset.from_list([{"input_ids": [1, 2], "loss_mask": [False, True]},
                              {"input_ids": [1, 2], "loss_mask": [True, False]}])
    filtered, _ = filter_training_records(data, 4, tmp_path)
    assert len(filtered) == 1
    bare = Dataset.from_list([{"input_ids": []}, {"input_ids": [1]}, {"input_ids": [1, 2]}])
    filtered, _ = filter_training_records(bare, 4, tmp_path)
    assert len(filtered) == 1


@pytest.mark.parametrize("mask,message", [
    ([0], "align"), ([0, -1], "nonnegative"), ([0, float("nan")], "finite"),
    ([0, float("inf")], "finite"), ([0, None], "null weight"),
])
def test_malformed_masks_remain_errors(tmp_path, mask, message):
    data = Dataset.from_list([{"input_ids": [1, 2], "loss_mask": mask}])
    with pytest.raises(ValueError, match=message):
        filter_training_records(data, 4, tmp_path)


def test_all_unsupervised_fails_before_loading_model(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("four_anchor.train")
    data = Dataset.from_list([{"input_ids": [1, 2], "loss_mask": [0, 0]}])
    monkeypatch.setattr(module, "load_records", lambda *a, **kw: data)
    def unexpected_model(*args, **kwargs):
        raise AssertionError("An unusable dataset should fail before model allocation")
    monkeypatch.setattr(module, "load_target", unexpected_model)
    cfg = module.TrainConfig("tiny", "unused", str(tmp_path), 63, [0], device="cpu")
    with pytest.raises(ValueError, match="No trainable rows"):
        module.train(cfg)


def test_six_rank_sampler_partitions_only_trainable_rows(tmp_path):
    data = Dataset.from_list([{"input_ids": [1, i + 2], "loss_mask": [0, i % 2]} for i in range(17)])
    filtered, _ = filter_training_records(data, 4, tmp_path)
    partitions = []
    for rank in range(6):
        sampler = DistributedSampler(filtered, num_replicas=6, rank=rank, shuffle=True, seed=42)
        partitions.append(list(sampler))
    assert len({len(part) for part in partitions}) == 1
    assert set(i for part in partitions for i in part) == set(range(len(filtered)))
    for part in partitions:
        assert all(filtered[i]["loss_mask"][1] > 0 for i in part)
