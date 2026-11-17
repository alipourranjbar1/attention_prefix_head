"""Verify that the method and optimizer remain exactly the pinned upstream code."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PORT_FILES = {
    "specforge/__init__.py", "specforge/core/__init__.py",
    "specforge/modeling/__init__.py", "specforge/modeling/draft/__init__.py",
    "specforge/modeling/target/__init__.py", "specforge/distributed.py",
    "scripts/train_dflash.py",
}


def verify(root=ROOT):
    original = json.loads((root / "UPSTREAM_SHA256.json").read_text())
    changes = []
    for relative, expected in original.items():
        path = root / "upstream" / relative
        if not path.is_file():
            raise RuntimeError(f"Missing upstream source: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            if relative not in ALLOWED_PORT_FILES:
                raise RuntimeError(f"Unexpected upstream method modification: {relative}")
            changes.append(relative)
    return changes


if __name__ == "__main__":
    changes = verify()
    print("Verified pinned DLoop model, CE objective, optimizer and LR scheduler unchanged.")
    print("Port-only source changes:", changes)
