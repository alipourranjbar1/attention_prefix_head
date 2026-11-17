"""Install the previously validated bias-free RMSNorm fallback for CANN 9.1."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys


def patch(source_root):
    path = Path(source_root) / "vllm_ascend/ops/layernorm.py"
    original = path.read_text()
    tree = ast.parse(original)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendRMSNorm")
    lines = original.splitlines(keepends=True)
    source = "".join(lines[cls.lineno - 1:cls.end_lineno])
    old = "            if enable_custom_op():\n"
    new = "            if self.bias is not None and enable_custom_op():\n"
    if new in source:
        print("Ascend RMSNorm CANN 9.1 fallback already installed")
        return None
    if source.count(old) != 1 or "torch_npu.npu_add_rms_norm(" not in source:
        raise ValueError("Unexpected AscendRMSNorm source; no changes made")
    replacement = source.replace(old, new, 1)
    updated = "".join(lines[:cls.lineno - 1]) + replacement + "".join(lines[cls.end_lineno:])
    compile(updated, str(path), "exec")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(path.name + ".bak_four_anchor_" + stamp)
    shutil.copy2(path, backup)
    path.write_text(updated)
    print("Installed Ascend RMSNorm fallback; backup:", backup)
    return backup


if __name__ == "__main__":
    patch(sys.argv[1])
