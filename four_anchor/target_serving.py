"""Launch only the hidden-state teacher with isolated graph settings."""
import json
import os
from pathlib import Path
import shlex
import sys


def settings(env):
    mode = env.get("TARGET_EXECUTION", "eager")
    if mode not in {"eager", "graph"}:
        raise ValueError("TARGET_EXECUTION must be eager or graph")
    extra = []
    child = dict(env)
    for key in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "ASCEND_LAUNCH_BLOCKING"):
        child.pop(key, None)
    child.update(VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_BREAKABLE_CUDAGRAPH="0",
                 VLLM_ASCEND_BALANCE_SCHEDULING="0")
    record = {"requested_execution": mode, "runtime_graph_verified": False,
              "scope": "vLLM hidden-state teacher only", "trainer_compile_environment_changed": False}
    if mode == "eager":
        extra.append("--enforce-eager")
    else:
        sizes = [int(x) for x in env.get("TARGET_GRAPH_SIZES", "128,256,512,1024,2048,3072").split(",")]
        if not sizes or sorted(set(sizes)) != sizes or not 1 <= sizes[0] <= sizes[-1] <= 9216:
            raise ValueError("TARGET_GRAPH_SIZES must be increasing unique positive sizes <= 9216")
        compilation = {"mode": 3, "cudagraph_mode": "PIECEWISE", "custom_ops": ["all"],
                       "cudagraph_capture_sizes": sizes, "max_cudagraph_capture_size": sizes[-1]}
        # Use the pinned CANN 9.1-compatible ACLGraph path. Do not enable
        # Npugraph_ex/static kernels or edit shared vLLM/Ascend dependencies.
        additional = {"enable_reduce_sample": False, "ascend_compilation_config": {
            "enable_npugraph_ex": False, "enable_static_kernel": False,
            "enable_super_kernel": False, "fuse_norm_quant": False,
            "fuse_qknorm_rope": False, "fuse_muls_add": False}}
        extra += ["--compilation-config", json.dumps(compilation), "--additional-config", json.dumps(additional),
                  "--max-num-batched-tokens", str(sizes[-1]), "--max-num-seqs", "32",
                  "--enable-chunked-prefill", "--cudagraph-metrics"]
        record.update(compilation_config=compilation, additional_config=additional,
                      max_num_batched_tokens=sizes[-1], max_num_seqs=32)
    return child, extra, record


def main():
    env, extra, record = settings(os.environ)
    output = Path(env["OUTPUT_DIR"])
    targets = env["TARGET_NPUS"].split(",")
    command = [sys.executable, str(Path(env["SPECULATORS_ROOT"]) / "scripts/launch_vllm.py"),
        "train", env["MODEL"], "--target-layer-ids", "2", "10", "18", "26", "34", "--include-last-layer",
        "--hidden-states-path", str(output / "hidden_states"), "--provenance-dir", str(output), "--",
        "--served-model-name", "four-anchor-teacher", "--host", "127.0.0.1",
        "--data-parallel-size", str(len(targets)), "--data-parallel-rpc-port", env["RPC_PORT"],
        "--api-server-count", "1", "--renderer-num-workers", "2", "--port", env["PORT"],
        "--max-model-len", str(int(env["SEQ_LENGTH"]) + 256), "--gpu-memory-utilization", "0.85", *extra]
    record["command"] = command
    (output / "target_execution.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Teacher launch: " + shlex.join(command), flush=True)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
