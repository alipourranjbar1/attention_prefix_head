import importlib.util
from pathlib import Path

import pytest


def helper():
    path = Path(__file__).parents[1] / "scripts/ensure_ascend_compat.py"
    spec = importlib.util.spec_from_file_location("compat_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compat_patch_is_scoped_idempotent_and_preserves_backup(tmp_path):
    path = tmp_path / "vllm_ascend/ops/layernorm.py"
    path.parent.mkdir(parents=True)
    original = '''# User's existing comment
class AscendRMSNorm:
    def forward_oot(self, x, residual):
        if residual is not None:
            if enable_custom_op():
                return custom(x)
            return torch_npu.npu_add_rms_norm(x)

class OtherNorm:
    def forward_oot(self, x, residual):
        if residual is not None:
            if enable_custom_op():
                return custom(x)
'''
    path.write_text(original)
    backup = helper().patch(tmp_path)
    assert backup.read_text() == original
    modified = path.read_text()
    assert modified.startswith("# User's existing comment")
    assert modified.count("if self.bias is not None and enable_custom_op():") == 1
    assert modified.count("            if enable_custom_op():") == 1
    assert helper().patch(tmp_path) is None
    assert path.read_text() == modified
    compile(modified, str(path), "exec")


def test_unknown_compat_source_is_not_overwritten(tmp_path):
    path = tmp_path / "vllm_ascend/ops/layernorm.py"
    path.parent.mkdir(parents=True)
    original = "class AscendRMSNorm:\n    pass\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="no changes"):
        helper().patch(tmp_path)
    assert path.read_text() == original
