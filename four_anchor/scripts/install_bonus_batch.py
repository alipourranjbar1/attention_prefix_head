#!/usr/bin/env python3
"""Install only into a NEW sibling worktree; never switch/edit the training tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASE = "01d7dea27438c6c0578215f51b19a7d89c843cea"
ALLOWED = {
    "README.md", "four_anchor/inference.py", "four_anchor/vllm_support.py",
    "four_anchor/vllm_ascend.py", "four_anchor/vllm_entry.py",
    "four_anchor/scripts/vllm_ascend.sh", "four_anchor/scripts/run_cached_benchmarks.py",
    "four_anchor/scripts/accepted_prefix_ascend.sh", "four_anchor/scripts/install_accepted_prefix.py",
    "four_anchor/ACCEPTED_PREFIX.md", "four_anchor/ACCEPTED_PREFIX_RUNTIME.json",
    "four_anchor/tests/test_accepted_prefix.py", "four_anchor/tests/test_accepted_prefix_install.py",
    "four_anchor/BONUS_ADDON.md", "four_anchor/BONUS_ADDON_RUNTIME.json",
    "four_anchor/scripts/bonus_addon_ascend.sh", "four_anchor/scripts/install_bonus_addon.py",
    "four_anchor/tests/test_bonus_addon.py", "four_anchor/tests/test_bonus_addon_install.py",
    'four_anchor/batch_inference.py',
    'four_anchor/vllm_batch_ascend.py',
    'four_anchor/vllm_batch_entry.py',
    'four_anchor/scripts/vllm_batch_ascend.sh',
    'four_anchor/scripts/bonus_batch_ascend.sh',
    'four_anchor/scripts/run_batched_benchmarks.py',
    'four_anchor/scripts/install_bonus_batch.py',
    'four_anchor/BONUS_BATCH.md',
    'four_anchor/BONUS_BATCH_RUNTIME.json',
    'four_anchor/tests/test_batch_inference.py',
    'four_anchor/tests/test_batch_entry.py',
    'four_anchor/tests/test_bonus_batch_install.py',
}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(root):
    """Read tracked files, index and branch; no refresh or status writes."""
    files = git(root, "ls-files", "-z").decode().split("\0")
    result = {}
    for name in files:
        if not name:
            continue
        p = root / name
        if p.is_symlink():
            result[name] = ("symlink", str(p.readlink()))
        elif p.is_file():
            result[name] = ("file", digest(p.read_bytes()), p.stat().st_mode)
        elif p.is_dir():  # submodule: read only its HEAD, never update it
            result[name] = ("directory",)
        else:
            result[name] = ("missing",)
    index = Path(git(root, "rev-parse", "--git-path", "index").decode().strip())
    if not index.is_absolute():
        index = root / index
    result["<index>"] = digest(index.read_bytes()) if index.exists() else None
    result["<head>"] = git(root, "symbolic-ref", "--short", "HEAD").decode().strip()
    return result


def install(package, source, destination, branch):
    package, source, destination = (Path(p).expanduser().resolve() for p in (package, source, destination))
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Destination must be a separate sibling, outside the training tree")
    if destination.exists():
        raise ValueError(f"Destination already exists; nothing was changed: {destination}")
    actual_root = Path(git(source, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if actual_root != source:
        raise ValueError("--training-root must name the repository root")
    if git(source, "rev-parse", BASE + "^{commit}").decode().strip() != BASE:
        raise ValueError("The saved training base commit is unavailable")
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest.get("base_commit") != BASE or set(manifest.get("files", {})) != ALLOWED:
        raise ValueError("Unexpected package base or file allowlist")
    payload = {}
    for name, info in manifest["files"].items():
        data = (package / "source" / name).read_bytes()
        if digest(data) != info["sha256"] or info["mode"] not in (420, 493):
            raise ValueError(f"Damaged package: {name}")
        payload[name] = data
    # Refuse branch collisions. No reset, checkout, stash, pull, submodule
    # initialization, dependency installation, or process termination occurs.
    git(source, "check-ref-format", "--branch", branch)
    existing = subprocess.run(["git", "-C", str(source), "show-ref", "--verify", "--quiet", "refs/heads/" + branch])
    if existing.returncode == 0:
        raise ValueError(f"Branch already exists: {branch}; choose a new --branch and destination")
    if existing.returncode != 1:
        raise ValueError("Could not check the destination branch")
    before = snapshot(source)
    subprocess.run(["git", "-C", str(source), "-c", "submodule.recurse=false", "-c", "core.hooksPath=/dev/null", "worktree", "add",
                    "-b", branch, str(destination), BASE], check=True)
    for name, data in payload.items():
        dest = destination / name
        if dest.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(manifest["files"][name]["mode"])
    after = snapshot(source)
    if before != after:
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        raise RuntimeError(f"Training tree changed during installation; inspect these paths: {changed}")
    subprocess.run(["git", "-C", str(destination), "diff", "--check"], check=True)
    print(f"Installed inference in: {destination}")
    print(f"Branch: {branch} (changes are ready for review; not committed or pushed)")
    print("Verified: original tracked files, index and branch are unchanged.")
    print("No NPU process was started or stopped. No training outputs were opened or written.")
    print(f"Guide: {destination / 'four_anchor/BONUS_BATCH.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--training-root")
    parser.add_argument("--destination")
    parser.add_argument("--branch", default="four-anchor-bonus20-c32-inference")
    parser.add_argument("--stage", action="store_true", help="Stage only the inference allowlist in this installed checkout")
    args = parser.parse_args()
    if args.stage:
        root = Path(__file__).resolve().parents[2]
        if not (root / "four_anchor/BONUS_BATCH_RUNTIME.json").is_file():
            parser.error("--stage must run from an installed bonus-addon checkout")
        if Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
            parser.error("Not the installed repository root")
        subprocess.run(["git", "-C", str(root), "add", "--", *sorted(ALLOWED)], check=True)
        return
    if not args.training_root or not args.destination:
        parser.error("--training-root and --destination are required for installation")
    install(args.package, args.training_root, args.destination, args.branch)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
