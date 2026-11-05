# Inference from four cached target anchors

Branch: `four-anchor-accepted-prefix-inference`.
Base: `01d7dea27438c6c0578215f51b19a7d89c843cea` on
`four-anchor-15-train-benchmarks`.
Inference version: `accepted_prefix19_v1`.

This is an **inference-only policy** for the existing format-3 `target_four`
checkpoint. It removes the extra target forward that materialized the pending
bonus. The existing checkpoint, model parameters, training objective, data,
training launchers, epoch evaluator and submodule revisions are unchanged.
Install into a separate worktree; do not copy the patch over a running training
checkout. Checkpoints and frozen benchmark prompts are read-only inputs.

## Two draft rounds, one target verification

Let the processed target prefix end at token **t10**. Its cached target states
include h7, h8, h9, h10. The target has already sampled **t11**, the pending
bonus/correction; it has **not processed t11** yet.

| Stage | Anchor tokens and states | Fifteen masked positions | Result used |
|---|---|---|---|
| Round one | t7–t10, actual target h7–h10 | t11–t25 | First-pass drafter states d11–d25 |
| Fix known slot | No new model call | Keep known t11 ID instead of its prediction | d11 remains a drafter state, never a fabricated target state |
| Round two, example restart t15 | t11–t14, drafter d11–d14 | t15–t29 | Refine from t15; extend coverage to t26 |
| Native verification | Normal target KV prefix through t10 | Target inputs: known t11 + candidates t12–t26 | Accept a prefix, then sample one new correction/bonus |
| Cache update | Retain h11 and states of accepted input candidates | Discard rejected suffix states | Next round starts from the last four processed tokens |

Each draft pass uses **4 anchors + 15 masks = 19 queries**. Exactly **15
speculative candidates** (t12–t26 in this example) go to native vLLM. Acceptance
including the single new correction/bonus is at most **16**; the four anchors
are excluded. The first-round window includes the already-known pending token;
round two supplies the final candidate beyond that first window. No twentieth
query is added and no candidate count is silently reduced to fourteen.

The restart is confidence-selected among the fourteen unknown first-pass
positions t12–t25 using the existing `boundary_scores` calculation. The known
t11 slot is always retained. This requires **two_pass** mode; one_pass is
rejected for this policy. A retained draft token is not target-accepted until
verification.

Round-two anchors use the original feature-source rule. For example:

| Restart | Four anchor states |
|---|---|
| t12 | target h8, h9, h10; drafter d11 |
| t13 | target h9, h10; drafter d11, d12 |
| t14 | target h10; drafter d11, d12, d13 |
| t15 or later | Four corresponding first-pass drafter states |

Prefill supplies the prompt's last four states; there is no need to generate
four startup tokens. Prompts shorter than four tokens use the existing masked
padding rule. If no speculative candidate is accepted, the pending token is
still processed by the normal target forward, so decoding progresses.

The newly sampled output bonus is **not** inserted into the feature cache.
Its actual target state becomes available when it is an input to the next
normal verification. Near the context limit, the existing adapter falls back
to ordinary target decoding rather than exceeding its position capacity.

## Relation to training

The same 19-position attention equations, weights, target-state projection,
drafter-state projection and four-anchor source rule are used. Training code
is byte-identical to the saved base. Training's first view predicts fifteen
unknown positions; this policy knows the first position's token ID from native
vLLM sampling and fixes that ID **after round one**. Its feature is the masked
first-pass drafter state. The restart is therefore restricted to positions
1–14 instead of 0–14. This is an inference policy adjustment, not a claim of
identical training and serving control flow. Acceptance must be measured.

No target h[bonus] is borrowed from the preceding token. No first-pass draft
state is labeled as a target state. No retraining or checkpoint conversion
is required. The previous materialized-bonus policy remains the default in
generic entrypoints; the new launcher explicitly selects `accepted_prefix`.

## Install without changing the training checkout

Unzip the supplied package and run `install.py` with the training root and a
new sibling destination. The installer pins the saved base commit, validates
every payload hash and an inference-only allowlist, and refuses an existing
destination or branch. It creates a worktree without initializing/updating
submodules. It verifies the original tracked files, index and branch stayed
unchanged. It never starts or stops processes, installs dependencies, or
opens training output files. Git's shared worktree/branch bookkeeping is the
only change in the original repository's metadata.

Use the existing `attention_prefix_ali` environment and already installed,
pinned vLLM/Ascend dependencies from `four-anchor-dflash-pipeline`. The new
launcher reads them; it does not patch them. The saved training branch and its
old runtime hash manifests are preserved as they were.

## Run on NPU 7 when it is free

The last supplied snapshot showed PID 2392590 on NPU 7. Let that job finish;
this package does not kill it or stop the training epoch evaluator. Check
`npu-smi info` before launching. Use a period without another evaluator scheduled
on NPU 7. For comparable speed, also measure a quiet-system run after training.

```bash
conda activate attention_prefix_ali
ROOT=/home/n84449292/a84384644/attention_head_forked/four-anchor-accepted-prefix-inference

# No model execution: check configuration or prepare all frozen prompts.
bash "$ROOT/four_anchor/scripts/accepted_prefix_ascend.sh" check
bash "$ROOT/four_anchor/scripts/accepted_prefix_ascend.sh" prepare

# Four GSM8K prompts, 128 output tokens, AR then speculative with exact parity.
EVAL_NPU=7 bash "$ROOT/four_anchor/scripts/accepted_prefix_ascend.sh" smoke

# After the smoke test passes: all eight cached benchmarks, up to 128 each,
# 2048 output tokens, matching AR baseline and speculative run for each.
EVAL_NPU=7 bash "$ROOT/four_anchor/scripts/accepted_prefix_ascend.sh" benchmarks
```

Defaults read epoch-one `checkpoint-022309` and the frozen `evaluation/prepared/plan.json`
under training run `target19_20261104_024656_3124841`. Override `DRAFT` or
`BENCHMARK_PLAN` if needed. Use `BENCHMARKS=gsm8k` with `benchmarks` to run just
the 128 GSM8K prompts. The frozen prompt hashes, target, limits and output
budget are checked. No datasets are downloaded. Both target and draft graphs
are off, concurrency is one, and decoding is greedy, matching the earlier
evaluation settings.

**New results are written only under the new inference checkout:**

```text
four-anchor-accepted-prefix-inference/output/accepted_prefix_<timestamp>_<pid>/
  summary.md                         # all-benchmark results table
  summary.csv
  suite_summary.json
  plan.json
  gsm8k/ar/{summary.json,per_prompt.jsonl,eval.log}
  gsm8k/two_pass/{summary.json,per_prompt.jsonl,eval.log}
  ...other benchmarks...
```

The smoke test instead writes `ar/` and `two_pass/` directly under its output
directory. `extra_target_forwards` is read from actual worker counters and must
be zero (including warmup); an unexpected extra call fails evaluation. Results
report acceptance, throughput including prefill, exact token parity, and
speedup against the fresh AR baseline. Accuracy is not measured. A parity
failure marks results provisional and fails that benchmark; do not present it
as a validated lossless speedup. No performance numbers are claimed for this
patch until it runs on the Ascend machine.

## Review and push the isolated branch

The installer leaves the new checkout's changes uncommitted for review.
Do this from the **new** directory:

```bash
cd /home/n84449292/a84384644/attention_head_forked/four-anchor-accepted-prefix-inference
git diff --stat
git add README.md four_anchor/inference.py four_anchor/vllm_support.py \
  four_anchor/vllm_ascend.py four_anchor/vllm_entry.py \
  four_anchor/scripts/vllm_ascend.sh four_anchor/scripts/run_cached_benchmarks.py \
  four_anchor/scripts/accepted_prefix_ascend.sh four_anchor/scripts/install_accepted_prefix.py \
  four_anchor/ACCEPTED_PREFIX.md four_anchor/ACCEPTED_PREFIX_RUNTIME.json \
  four_anchor/tests/test_accepted_prefix.py four_anchor/tests/test_accepted_prefix_install.py
git diff --cached --check
git commit -m "Add isolated 19-position inference using cached accepted-prefix states"
git push -u origin four-anchor-accepted-prefix-inference
```

Do not add checkpoint weights, hidden-state caches or full output directories
to Git. After evaluation, copy the small result table and summary JSON into
a new `four_anchor/results/` subdirectory if you want to publish measurements.

## Validation

CPU tests cover every restart position, short-prefix padding, four-state
selection and alignment, 19-position equations matching the model, exactly
15 returned candidates, all 0–15 acceptance counts, discarded suffix states,
real tiny-Qwen greedy parity and output budgets, measured extra-call guards,
frozen prompts for all eight benchmarks, and installer isolation.
Ascend kernels, paged KV integration and Qwen3-4B performance still require
the NPU smoke test. No server process was controlled from the development
environment.
