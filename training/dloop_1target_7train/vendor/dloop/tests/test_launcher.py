import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from ascend_support.verify_sources import ROOT


@pytest.mark.parametrize("toolkit",["cann","ascend-toolkit"])
@pytest.mark.parametrize("mode",["pretrained","scratch"])
@pytest.mark.parametrize("packed",[False,True])
def test_launcher_environment_order_native_recipe_and_no_old_imports(tmp_path,toolkit,mode,packed):
    cann=tmp_path/"CANN"
    (cann/toolkit).mkdir(parents=True)
    (cann/toolkit/"set_env.sh").write_text("export CANN_READY=1\n")
    atb=cann/"nnal/atb";atb.mkdir(parents=True)
    (atb/"set_env.sh").write_text(
        '[[ "$1" == "--cxx_abi=1" ]] || return 10\n'
        'export ATB_READY=1 ATB_HOME_PATH="$CANN_ROOT/nnal/atb"\n')
    bindir=tmp_path/"bin";bindir.mkdir()
    stub=bindir/"python"
    stub.write_text(f"#!{sys.executable}\n"+'''
import json,os,sys
from pathlib import Path
a=sys.argv[1:]
assert os.environ.get("CANN_READY")=="1"
if a[0]=="-c":
    assert "_GLIBCXX_USE_CXX11_ABI" in a[1]
    print(1);sys.exit(0)
assert os.environ.get("ATB_READY")=="1"
assert "forbidden_old_project" not in os.environ["PYTHONPATH"]
if a[:2]==["-m","ascend_support.preflight"]:
    if "--resolve-draft" in a:
        if os.environ["INIT_MODE"] == "scratch":
            assert not os.environ.get("DRAFT_INIT")
            print(str(Path.cwd()/"configs/qwen3_4b_dflash_scratch"))
        else:
            print(os.environ["DRAFT_INIT"])
    else:
        Path(os.environ["OUTPUT_DIR"]).mkdir()
elif a[:2]==["-m","ascend_support.packed_data"]:
    assert not Path(os.environ["OUTPUT_DIR"]).exists()
    Path(a[a.index("--output")+1]).write_text(json.dumps({"index_file":"/test/index.npz","steps_per_epoch":25}))
elif a[0]=="-":
    sys.argv=a
    exec(compile(sys.stdin.read(),"launcher_record","exec"))
elif a[:2]==["-m","torch.distributed.run"]:
    Path(os.environ["TEST_CALLS"]).write_text(json.dumps(a))
else:
    raise RuntimeError(a)
''')
    stub.chmod(0o755)
    env=dict(os.environ)
    for key in ("RESUME","OUTPUT_DIR","MAX_STEPS","CHECK_ONLY","ACTIVATION_CHECKPOINTING","INIT_MODE","DRAFT_INIT","HEAD_LOSS_CHUNK_SIZE","PACKED_TOKEN_BUDGET","PACKING_CACHE_DIR"):
        env.pop(key,None)
    data=tmp_path/"data";data.mkdir()
    calls=tmp_path/"calls.json"
    env.update(PATH=f"{bindir}:{env['PATH']}",CONDA_PREFIX=str(tmp_path/"conda"),
        CANN_ROOT=str(cann),BASE=str(tmp_path),MODEL=str(tmp_path/"target"),
        DATASET=str(data),DRAFT_INIT=str(tmp_path/"regular-dflash"),
        OUTPUT_DIR=str(tmp_path/"result"),TEST_CALLS=str(calls),
        PYTHONPATH="forbidden_old_project",LOSS_FN='{"ce":0.1,"tv":0.9}',
        REFERENCE_CONFIG="forbidden_old_project/train_config.json",
        MAX_ANCHORS="1536",NPUS="0,1,2,3,4,5,6,7",EPOCHS="1",
        MAX_LENGTH="4096",NUM_ANCHORS="256",LR="0.0006")
    env["INIT_MODE"]=mode
    if mode == "scratch":
        env.pop("DRAFT_INIT")
        env.update(MAX_LENGTH="9216",NUM_ANCHORS="768",HEAD_LOSS_CHUNK_SIZE="128",ACTIVATION_CHECKPOINTING="1")
    if packed:
        env.update(PACKED_TOKEN_BUDGET="9216",MAX_LENGTH="3072",NUM_ANCHORS="768",
                   PACKING_CACHE_DIR=str(tmp_path/"packing-cache"))
    result=subprocess.run(["bash",str(ROOT/"train_ascend.sh")],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    a=json.loads(calls.read_text())
    for option,value in [("--num-epochs","1"),("--num-anchors",env["NUM_ANCHORS"]),
        ("--max-length",env["MAX_LENGTH"]),
        ("--loss-type","dflash"),("--loss-decay-gamma","7.0"),
        ("--dloop-max-stage","3"),("--position-offset-max","28672"),
        ("--target-model-backend","hf"),("--attention-backend","sdpa")]:
        assert a[a.index(option)+1]==value
    assert "--nproc_per_node=8" in a
    assert "--saved-tokenized-data" in a
    if packed:
        assert a[a.index("--packed-token-budget")+1] == "9216"
        assert a[a.index("--packing-index")+1] == "/test/index.npz"
        assert json.loads((tmp_path/"result/packing_plan.json").read_text())["steps_per_epoch"] == 25
    else:
        assert "--packed-token-budget" not in a
    assert not any("forbidden_old_project" in arg for arg in a)
    assert not any("loss_fn" in arg or "tv" == arg for arg in a)
    if mode == "scratch":
        assert "--scratch" in a and "--draft-config-path" in a
        assert "--draft-init-path" not in a
        assert a[a.index("--head-loss-chunk-size")+1]=="128"
        assert "--activation-checkpointing" in a
    else:
        assert "--scratch" not in a and "--draft-init-path" in a
