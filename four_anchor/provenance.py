from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys


def git(*args, cwd=None):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path, stage, settings):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ["torch", "torch-npu", "transformers", "speculators", "datasets", "safetensors"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
            "python": platform.python_version(), "packages": packages,
            "git_sha": git("rev-parse", "HEAD", cwd=root),
            "speculators_sha": git("rev-parse", "HEAD", cwd=root / "speculators"),
            "settings": settings,
            "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "four_anchor").rglob("*.py"))}}
    (path / f"{stage}_command.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    (path / f"{stage}_command.txt").write_text(json.dumps(info, indent=2, default=str) + "\n")
    (path / f"{stage}.patch").write_text(git("diff", "HEAD", cwd=root) or "")
    (path / "speculators.patch").write_text(git("diff", "HEAD", cwd=root / "speculators") or "")


def checkpoint_hashes(path):
    path = Path(path)
    files = [path / "model.safetensors", path / "four_anchor_config.json"]
    (path / "checkpoint_sha256.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
