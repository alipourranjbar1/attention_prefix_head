#!/usr/bin/env python3
"""Train/evaluate an acceptance-length head from verification records."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shlex
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def implementation():
    model_directory = (
        Path(__file__).resolve().parents[2]
        / "src/speculators/models/dflash_prefix"
    )

    def load_module(name, filename):
        path = model_directory / filename

        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {path}")

        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    head_module = load_module(
        "acceptance_head_local",
        "acceptance_head.py",
    )

    loss_module = load_module(
        "acceptance_loss_local",
        "acceptance_loss.py",
    )

    return head_module, loss_module


class Records(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        record = torch.load(
            self.paths[index],
            map_location="cpu",
            weights_only=True,
        )

        features = record["features"]
        label = torch.tensor(
            record["accepted_length"],
            dtype=torch.long,
        )

        return features, label


def inspect_records(trace_directory):
    """
    Expected directory layout:

        traces/
            collection_run/
                manifest.json
                000000001.pt
                000000002.pt

    Each record contains:
        features:        Tensor [K, hidden_size + 5]
        accepted_length: Python integer in 0...K
        prompt_key:      Stable string identifying the original prompt

    Manifests must match across collection runs and include:
        proposals: K

    The collector should also record checkpoint identities and feature schema.
    """
    manifests = sorted(trace_directory.rglob("manifest.json"))

    if not manifests:
        raise ValueError(
            f"No manifest.json found under {trace_directory}. "
            "Collect verification records before training."
        )

    identity = json.loads(manifests[0].read_text())

    num_proposals = identity.get("proposals")
    if type(num_proposals) is not int or num_proposals < 1:
        raise ValueError(
            "manifest.json must contain a positive integer 'proposals'."
        )

    train_paths = []
    validation_paths = []
    feature_shape = None

    for manifest in manifests:
        current_identity = json.loads(manifest.read_text())

        if current_identity != identity:
            raise ValueError(
                "Mixed checkpoint identities or feature schemas. "
                "Use separate trace directories."
            )

        for path in sorted(manifest.parent.glob("*.pt")):
            record = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            required = {"features", "accepted_length", "prompt_key"}
            if not required.issubset(record):
                raise ValueError(f"Missing record fields: {path}")

            features = record["features"]
            label = record["accepted_length"]
            prompt_key = record["prompt_key"]

            if not isinstance(features, torch.Tensor):
                raise TypeError(f"features must be a tensor: {path}")

            if features.ndim != 2:
                raise ValueError(
                    f"Expected features with shape [K, H + 5]: {path}"
                )

            if feature_shape is None:
                feature_shape = tuple(features.shape)

            if tuple(features.shape) != feature_shape:
                raise ValueError(f"Inconsistent feature dimensions: {path}")

            if features.shape[0] != num_proposals:
                raise ValueError(
                    f"Feature proposal count disagrees with manifest: {path}"
                )

            if features.shape[1] <= 5:
                raise ValueError(f"Invalid feature width: {path}")

            if not bool(torch.isfinite(features.float()).all()):
                raise ValueError(f"Nonfinite features: {path}")

            if type(label) is not int or not 0 <= label <= num_proposals:
                raise ValueError(f"Invalid accepted_length: {path}")

            if not isinstance(prompt_key, str) or not prompt_key:
                raise ValueError(f"Invalid prompt_key: {path}")

            # Stable prompt-level split: approximately 10% validation.
            # Repeated prompts stay together across collection runs.
            digest = hashlib.sha256(prompt_key.encode()).hexdigest()
            is_validation = int(digest[:8], 16) % 10 == 0

            if is_validation:
                validation_paths.append(path)
            else:
                train_paths.append(path)

    if feature_shape is None:
        raise ValueError("Manifests exist, but no .pt records were found.")

    return identity, feature_shape, train_paths, validation_paths


def run_epoch(
    head,
    loader,
    device,
    loss_function,
    args,
    optimizer=None,
):
    training = optimizer is not None
    head.train(training)

    totals = {}
    example_count = 0

    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device=device, dtype=torch.long)

            logits = head(features)

            loss, metrics = loss_function(
                logits,
                labels,
                under_weight=args.under_weight,
                over_weight=args.over_weight,
                cost_weight=args.cost_weight,
            )

            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite loss encountered.")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    head.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            batch_size = labels.numel()
            example_count += batch_size

            # Transfer logging metrics together.
            names = list(metrics)
            values = torch.stack(
                [metrics[name].detach() for name in names]
            ).cpu().tolist()

            for name, value in zip(names, values):
                totals[name] = (
                    totals.get(name, 0.0) + value * batch_size
                )

    if example_count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    result = {
        name: value / example_count
        for name, value in totals.items()
    }
    result["examples"] = example_count

    return result


def save_checkpoint(path, head, identity, epoch, metrics, args):
    checkpoint = {
        "architecture": dict(head.architecture),
        "identity": identity,
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in head.state_dict().items()
        },
        "epoch": epoch,
        "validation_metrics": metrics,
        "loss_settings": {
            "under_weight": args.under_weight,
            "over_weight": args.over_weight,
            "cost_weight": args.cost_weight,
        },
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--projection", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)

    parser.add_argument("--under-weight", type=float, default=3.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    parser.add_argument("--cost-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--evaluate",
        type=Path,
        help="Evaluate a saved head on the validation prompt split.",
    )

    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.projection,
        args.width,
    ) < 1:
        parser.error(
            "epochs, batch-size, projection and width must be positive."
        )

    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive.")

    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative.")

    weights = (
        args.under_weight,
        args.over_weight,
        args.cost_weight,
    )

    if any(not math.isfinite(value) or value < 0 for value in weights):
        parser.error("Loss weights must be finite and nonnegative.")

    if args.under_weight <= args.over_weight:
        parser.error("under-weight must exceed over-weight.")

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    head_module, loss_module = implementation()

    identity, shape, train_paths, validation_paths = inspect_records(
        args.traces
    )

    if not validation_paths:
        parser.error(
            "No validation examples. Collect more distinct prompts; "
            "approximately 10% of prompt hashes are used for validation."
        )

    if not args.evaluate and not train_paths:
        parser.error("No training examples. Collect more distinct prompts.")

    architecture = {
        "hidden_size": shape[1] - 5,
        "num_proposals": shape[0],
        "projection_size": args.projection,
        "mlp_hidden_size": args.width,
    }

    saved = None

    if args.evaluate:
        saved = torch.load(
            args.evaluate,
            map_location="cpu",
            weights_only=True,
        )

        if saved["identity"] != identity:
            parser.error(
                "Evaluation head and trace checkpoint identities differ."
            )

        architecture = saved["architecture"]

        expected_keys = {
            "hidden_size",
            "num_proposals",
            "projection_size",
            "mlp_hidden_size",
        }

        if set(architecture) != expected_keys:
            parser.error(
                "This head checkpoint uses an incompatible architecture "
                "format. Use a checkpoint trained with this manual version."
            )

        if (
            architecture["hidden_size"] != shape[1] - 5
            or architecture["num_proposals"] != shape[0]
        ):
            parser.error("Head architecture does not match trace features.")

    head = head_module.AcceptedLengthHead(
        **architecture
    ).to(device)

    if saved is not None:
        head.load_state_dict(
            saved["state_dict"],
            strict=True,
        )

    # Use a new directory for every training/evaluation run.
    if args.output.exists():
        parser.error(
            f"Output directory already exists: {args.output}. "
            "Choose a new output directory."
        )

    args.output.mkdir(parents=True)

    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration.update(
        architecture=architecture,
        identity=identity,
        torch_version=str(torch.__version__),
        train_records=len(train_paths),
        validation_records=len(validation_paths),
        split="sha256(prompt_key)[:8] modulo 10 == 0 is validation",
    )

    (args.output / "training_config.json").write_text(
        json.dumps(configuration, indent=2)
    )

    (args.output / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )

    (args.output / "split.json").write_text(
        json.dumps(
            {
                "train": [str(path) for path in train_paths],
                "validation": [
                    str(path) for path in validation_paths
                ],
            },
            indent=2,
        )
    )

    validation_loader = DataLoader(
        Records(validation_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if saved is not None:
        report = run_epoch(
            head,
            validation_loader,
            device,
            loss_module.acceptance_boundary_loss,
            args,
        )

        (args.output / "evaluation.json").write_text(
            json.dumps(report, indent=2)
        )

        print(json.dumps(report, indent=2), flush=True)
        return

    train_loader = DataLoader(
        Records(train_paths),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in head.parameters()
        if parameter.requires_grad
    )

    print(
        f"Training examples: {len(train_paths)}\n"
        f"Validation examples: {len(validation_paths)}\n"
        f"Draft proposals K: {shape[0]}\n"
        f"Hidden size: {shape[1] - 5}\n"
        f"Trainable head parameters: {parameter_count:,}",
        flush=True,
    )

    best_score = (float("inf"), float("inf"))

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            head,
            train_loader,
            device,
            loss_module.acceptance_boundary_loss,
            args,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            head,
            validation_loader,
            device,
            loss_module.acceptance_boundary_loss,
            args,
        )

        report = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }

        print(json.dumps(report), flush=True)

        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")

        # Prefer lower asymmetric cost; break ties with cross-entropy.
        score = (
            validation_metrics["decision_cost"],
            validation_metrics["cross_entropy"],
        )

        if score < best_score:
            best_score = score

            save_checkpoint(
                args.output / "best.pt",
                head,
                identity,
                epoch,
                validation_metrics,
                args,
            )

        save_checkpoint(
            args.output / "last.pt",
            head,
            identity,
            epoch,
            validation_metrics,
            args,
        )

    print(
        f"Training finished. Best checkpoint: {args.output / 'best.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()