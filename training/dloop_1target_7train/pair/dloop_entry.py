"""Run the pinned Ascend trainer with optional exact attention acceleration."""
import os
import runpy
from pathlib import Path
import torch_npu  # initialize the accelerator before model imports
from ascend_support.verify_sources import verify, ROOT

verify()
if os.environ.get("DLOOP_REMOTE_TEACHER_CONFIG"):
    from .remote_dloop import install as install_remote
    install_remote(os.environ["DLOOP_REMOTE_TEACHER_CONFIG"])
    print("DLoop: separate HF target services; original RoPE offsets preserved", flush=True)
if os.environ.get("FAST_DLOOP", "1") == "1":
    from .fast_attention import install
    install()
    print("DLoop: exact block-isolated SDPA enabled; carrier gradients preserved", flush=True)
runpy.run_path(str(ROOT / "upstream/scripts/train_dflash.py"), run_name="__main__")
