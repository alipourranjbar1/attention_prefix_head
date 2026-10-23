"""Collect full-block verification examples for acceptance-head training."""

import hashlib
import json
import uuid
from pathlib import Path

import torch

from .acceptance_features import FEATURE_SCHEMA


def checkpoint_fingerprint(directory):
    """Hash checkpoint configuration and safetensors weights once at startup."""
    directory = Path(directory).expanduser().resolve(strict=True)

    config_path = directory / "config.json"
    weight_paths = sorted(directory.glob("*.safetensors"))

    if not config_path.is_file() or not weight_paths:
        raise ValueError(
            f"Expected config.json and safetensors weights in {directory}"
        )

    digest = hashlib.sha256()

    for path in [config_path, *weight_paths]:
        # Hash each file independently, then combine names and hashes.
        file_digest = hashlib.sha256()

        with path.open("rb") as stream:
            while True:
                chunk = stream.read(8 * 1024 * 1024)
                if not chunk:
                    break
                file_digest.update(chunk)

        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(file_digest.digest())

    return digest.hexdigest()


class AcceptanceCollector:
    """
    Collect examples from full-block, greedy target verification.

    Required caller behavior:
        1. Record the actual draft and its corresponding features.
        2. Verify the complete draft block.
        3. Pass the verifier's generated tokens BEFORE user-facing trimming.
        4. Generated tokens must contain accepted draft tokens followed by
           exactly one target correction/bonus token.

    This initial collector is intended for synchronous, batch-one serving.
    """

    def __init__(
        self,
        trace_directory,
        draft_checkpoint,
        target_checkpoint,
        num_proposals,
        eos_token_ids=None,
    ):
        if type(num_proposals) is not int or num_proposals < 1:
            raise ValueError("num_proposals must be a positive integer.")

        self.num_proposals = num_proposals
        self.pending = {}
        self.round_counter = 0
        self.saved_count = 0
        self.skipped_count = 0

        target_checkpoint = Path(target_checkpoint).expanduser().resolve(
            strict=True
        )

        if eos_token_ids is None:
            target_config = json.loads(
                (target_checkpoint / "config.json").read_text()
            )
            eos_token_ids = target_config.get("eos_token_id")

        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        self.eos_token_ids = set(eos_token_ids)

        self.identity = {
            "schema": FEATURE_SCHEMA,
            "proposals": num_proposals,
            "draft_sha256": checkpoint_fingerprint(draft_checkpoint),
            "target_sha256": checkpoint_fingerprint(target_checkpoint),
            "decoding": "greedy_default_penalties_v1",
            "eos_token_ids": sorted(self.eos_token_ids),
        }

        # Every server session writes to a separate directory.
        self.directory = (
            Path(trace_directory).expanduser().resolve()
            / uuid.uuid4().hex
        )
        self.directory.mkdir(parents=True, exist_ok=False)

        (self.directory / "manifest.json").write_text(
            json.dumps(self.identity, indent=2) + "\n"
        )

    @torch.no_grad()
    def record_proposal(
        self,
        request_id,
        prompt_token_ids,
        draft_token_ids,
        features,
    ):
        """
        Save a pending draft until its verification result arrives.

        Args:
            request_id: Unique request identifier.
            prompt_token_ids: Original prompt IDs as a CPU list.
            draft_token_ids: Actual K proposed token IDs as a CPU list.
            features: Tensor [K, hidden_size + 5].
        """
        if prompt_token_ids is None:
            raise ValueError("Tokenized prompts are required for collection.")

        tokens = [int(token) for token in draft_token_ids]

        if len(tokens) != self.num_proposals:
            raise ValueError(
                "Collection requires the complete draft block."
            )

        if (
            features.ndim != 2
            or features.shape[0] != self.num_proposals
            or features.shape[1] <= 5
        ):
            raise ValueError(
                "Expected features with shape [K, hidden_size + 5]."
            )

        # Clone so reused accelerator/graph buffers cannot change the record.
        cpu_features = features.detach().to(
            device="cpu",
            dtype=torch.bfloat16,
        ).clone()

        if not bool(torch.isfinite(cpu_features.float()).all()):
            raise ValueError("Cannot collect nonfinite features.")

        prompt = [int(token) for token in prompt_token_ids]
        prompt_key = hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode()
        ).hexdigest()

        # Batch-one synchronous collection has only one outstanding draft.
        # Remove abandoned records when a request finishes without verification.
        self.skipped_count += len(self.pending)
        self.pending.clear()

        self.round_counter += 1

        self.pending[str(request_id)] = {
            "features": cpu_features,
            "draft_token_ids": tokens,
            "prompt_key": prompt_key,
            "request_id": str(request_id),
            "round_id": self.round_counter,
        }

    def record_verification(
        self,
        request_id,
        scheduled_draft_ids,
        generated_token_ids,
        *,
        discarded=False,
    ):
        """
        Attach a verified accepted length to the matching pending draft.

        All token arguments must be CPU lists.

        Example:
            Draft:     [10, 11, 12]
            Generated: [10, 11, 99]
            Label:     2

        Here 99 is a target correction token, not an EOS token.
        """
        example = self.pending.pop(str(request_id), None)

        if example is None:
            return False

        if discarded:
            self.skipped_count += 1
            return False

        scheduled = [int(token) for token in scheduled_draft_ids]
        generated = [int(token) for token in generated_token_ids]

        # Reject truncated, stale, or otherwise changed proposal blocks.
        if (
            len(scheduled) != self.num_proposals
            or scheduled != example["draft_token_ids"]
        ):
            self.skipped_count += 1
            return False

        # Avoid using EOS-censored rounds as complete boundary observations.
        if any(token in self.eos_token_ids for token in generated):
            self.skipped_count += 1
            return False

        # Standard greedy speculative verification emits A accepted
        # draft tokens and one target correction/bonus token.
        accepted_length = len(generated) - 1

        if not 0 <= accepted_length <= self.num_proposals:
            self.skipped_count += 1
            return False

        if generated[:accepted_length] != scheduled[:accepted_length]:
            self.skipped_count += 1
            return False

        # At an actual greedy rejection, the correction must differ
        # from the rejected draft token.
        if (
            accepted_length < self.num_proposals
            and generated[accepted_length] == scheduled[accepted_length]
        ):
            self.skipped_count += 1
            return False

        example["accepted_length"] = accepted_length

        destination = (
            self.directory / f"{example['round_id']:09d}.pt"
        )
        temporary = destination.with_suffix(".pt.tmp")

        torch.save(example, temporary)
        temporary.replace(destination)

        self.saved_count += 1
        return True