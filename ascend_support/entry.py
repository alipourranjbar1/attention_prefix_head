"""Initialize Ascend before running NAVER's training entry point."""
import runpy
import sys

from .verify_sources import ROOT, verify

verify()
import torch_npu  # noqa: E402,F401; registers NPU/HCCL before upstream imports

script = ROOT / "upstream/scripts/train_dflash.py"
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
