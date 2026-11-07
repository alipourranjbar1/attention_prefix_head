"""Integration checks: the installer may edit only a NEW inference checkout."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def installer():
    path = Path(__file__).parents[1] / "scripts/install_bonus_addon.py"
    spec = importlib.util.spec_from_file_location("installer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def package(installer, tmp_path):
    root = Path(__file__).parents[2]
    package = tmp_path / "package"
    files = {}
    for name in installer.ALLOWED:
        data = (root / name).read_bytes()
        path = package / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "mode": 493 if name.endswith(".sh") else 420}
    (package / "manifest.json").write_text(json.dumps({"base_commit": installer.BASE, "files": files}))
    return package


def test_install_creates_sibling_without_editing_source_index_branch_or_outputs(installer, package, tmp_path):
    repo = Path(__file__).parents[2]
    training = tmp_path / "training"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(repo), str(training)], check=True)
    subprocess.run(["git", "-C", str(training), "checkout", "-b", "training", installer.BASE], check=True)
    # Even dirty training files and untracked outputs must survive untouched.
    (training / "four_anchor/train.py").write_text("# running training source marker\n")
    checkpoint = training / "output/active/checkpoints/model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint sentinel")
    before = installer.snapshot(training)
    destination = tmp_path / "inference"
    installer.install(package, training, destination, "isolated-inference")
    assert installer.snapshot(training) == before
    assert checkpoint.read_bytes() == b"checkpoint sentinel"
    assert installer.git(destination, "branch", "--show-current").strip() == b"isolated-inference"
    assert installer.git(training, "branch", "--show-current").strip() == b"training"
    changed = set(installer.git(destination, "diff", "--name-only").decode().splitlines())
    untracked = set(installer.git(destination, "ls-files", "--others", "--exclude-standard").decode().splitlines())
    assert changed | untracked == installer.ALLOWED
    assert not (destination / "output").exists()
    # No training-source patch is present in the new checkout either.
    actual = installer.git(destination, "diff", installer.BASE, "--", "four_anchor/train.py",
                           "four_anchor/model.py", "four_anchor/epoch_benchmarks.py")
    assert actual == b""
    with pytest.raises(ValueError, match="already exists"):
        installer.install(package, training, destination, "isolated-inference")
    with pytest.raises(ValueError, match="sibling"):
        installer.install(package, training, training / "inference", "nested-inference")


def test_training_file_in_package_is_rejected_before_any_worktree_write(installer, package, tmp_path):
    repo = Path(__file__).parents[2]
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["files"]["four_anchor/train.py"] = {"sha256": "bad", "mode": 420}
    (package / "manifest.json").write_text(json.dumps(manifest))
    dest = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="allowlist"):
        installer.install(package, repo, dest, "should-never-be-created")
    assert not dest.exists()


def test_corrupt_patch_is_rejected_before_any_worktree_write(installer, package, tmp_path):
    repo = Path(__file__).parents[2]
    (package / "source/four_anchor/inference.py").write_text("bad patch")
    with pytest.raises(ValueError, match="Damaged package"):
        installer.install(package, repo, tmp_path / "corrupt", "should-never-be-created")
    assert not (tmp_path / "corrupt").exists()
