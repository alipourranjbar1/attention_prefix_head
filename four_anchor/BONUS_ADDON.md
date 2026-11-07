# Bonus embedding in round one; four drafter states in round two

Branch: `four-anchor-bonus20-inference`.
Version: `bonus_addon20_draft4_v1`.
Checkpoint contract: existing format-3 `target_four`, four anchors, fifteen
trained proposal slots (default: `checkpoint-022309`). No retraining,
checkpoint rewriting, dependency installation or training-file edits occur.

## Exact token and hidden-state layout

The processed prefix ends at t10. Target states h7–h10 already exist. The
target has sampled bonus b11, but has not processed it; actual h(b11) does
not exist yet. All indexing below is by token position, not token ID value.

| Stage | Query positions | Anchor feature source | Output used |
|---|---:|---|---|
| Round one | t7,t8,t9,t10 + b11 + masks t12–t26 = **20** | h7–h10 from target; **no added feature at b11** | Save all 20 normalized drafter outputs; predict 15 candidates t12–t26 |
| Select restart | No model call | Existing confidence boundary scores over only the 15 unknown candidates | Keep the predicted prefix; re-draft from the selected first-error position |
| Round two, example restart t20 | t16,t17,t18,t19 + masks t20–t34 = **19** | **d16,d17,d18,d19**, all from round one, through the trained draft projection | Replace t20–t26; predictions beyond t26 are not verified |
| Merge | Exactly t12–t26 = **15 candidates** | No target call between draft rounds | Round-one prefix + round-two suffix |
| Native vLLM verification | b11 + t12–t26 = **16 target inputs** | Normal target forward, once | Accept a candidate prefix and emit one new correction/bonus |
| Cache update | Processed, accepted inputs only | Actual target states | Discard rejected suffix; the newly sampled bonus has no target state yet |

The initial four anchors and their target features remain unchanged. The
bonus token is a real input embedding in round one, not an ID inserted after
drafting. Target prefix features are still cached as attention K/V in **both**
rounds. The rule "four drafter states in round two" applies to its four query
anchors, not to replacing the entire context K/V cache.

All four round-two anchor features come from the first drafter pass, including
the outputs at original past-anchor positions when the restart is early:

| Restart | Four anchor tokens | Four anchor states |
|---|---|---|
| t12 | t8,t9,t10,b11 | d8,d9,d10,d11 |
| t13 | t9,t10,b11,t12 | d9,d10,d11,d12 |
| t14 | t10,b11,t12,t13 | d10,d11,d12,d13 |
| t15 | b11,t12,t13,t14 | d11,d12,d13,d14 |
| t20 | t16,t17,t18,t19 | d16,d17,d18,d19 |
| t26 | t22,t23,t24,t25 | d22,d23,d24,d25 |

There is **no extra bonus slot in round two**. The existing bonus may be one
of its four preceding tokens. These retained proposals are not yet
target-accepted; only the final verification decides acceptance. Prefixes
shorter than four tokens use masked padding, never invented hidden states.

## Counts and acceptance

- First draft query length: **20** (4 + 1 + 15).
- Second draft query length: **19** (4 + 15).
- Speculative candidates: **15** in one-pass and two-pass modes.
- Verification block: **16** (known bonus + 15 candidates).
- Acceptance including correction/bonus: **1 + accepted_candidates**, at most **16**.
- The four conditioning anchors are **never** added to acceptance.
- The previously known bonus is not counted again as a newly accepted draft
  candidate. Native vLLM records accepted candidates plus the new output bonus;
  this is the standard 16-token acceptance convention, before EOS/budget truncation.
- Extra target forwards for bonus materialization: **zero**, checked by worker counters.

This is a new inference policy for the existing checkpoint, **not identical
to training**. Round one has one extra known-token query, and early round-two
restarts use drafter outputs even at past-anchor positions where training used
target features. The architecture supports these shapes without altering
weights. Acceptance and speed still need measurement; neither improvement nor
unchanged acceptance is guaranteed. The previous accepted-prefix experiment
also had one of four Qwen3-4B smoke prompts fail exact greedy parity; this new
policy needs its own NPU validation and does not claim to have resolved that.

## Install into a separate sibling checkout

Place `four-anchor-bonus20-inference.zip` in
`/home/n84449292/a84384644/attention_head_forked`, then:

```bash
conda activate attention_prefix_ali
(
  set -euo pipefail
  FORK=/home/n84449292/a84384644/attention_head_forked
  package_dir="$(mktemp -d)"
  unzip -q "$FORK/four-anchor-bonus20-inference.zip" -d "$package_dir"
  python "$package_dir/four-anchor-bonus20-inference/install.py" \
    --training-root "$FORK/four-anchor-15-train-benchmarks" \
    --destination "$FORK/four-anchor-bonus20-inference"
)
```

The installer creates a new branch from saved training commit
`01d7dea27438c6c0578215f51b19a7d89c843cea`. It refuses existing destinations or
branch names, validates the payload hashes and inference-only allowlist, and
checks the original tracked files, index and branch are unchanged. It does
not switch the training branch, update submodules, open checkpoints, or start
or stop processes. The original Git repository gains only worktree/branch
bookkeeping. The installed changes are uncommitted and ready for review.

## Run on NPU 7

The launcher reuses the existing `attention_prefix_ali` environment and pinned
vLLM/Ascend/Speculators checkout at `four-anchor-dflash-pipeline`. Both graphs
are off, concurrency is one, and decoding is greedy. Launch when NPU 7 is
available; it does not stop another job.

```bash
ROOT=/home/n84449292/a84384644/attention_head_forked/four-anchor-bonus20-inference

# No model execution: validates checkpoint/config/runtime.
bash "$ROOT/four_anchor/scripts/bonus_addon_ascend.sh" check

# Four GSM8K prompts, 128 output tokens, speculative only.
EVAL_NPU=7 bash "$ROOT/four_anchor/scripts/bonus_addon_ascend.sh" smoke

# 128 GSM8K prompts, 2048 output tokens, speculative only.
EVAL_NPU=7 bash "$ROOT/four_anchor/scripts/bonus_addon_ascend.sh" evaluate

# All eight frozen cached benchmarks, up to 128 each, speculative only.
EVAL_NPU=7 bash "$ROOT/four_anchor/scripts/bonus_addon_ascend.sh" benchmarks
```

The default checkpoint is read from
`four-anchor-15-train-benchmarks/output/target19_20261104_024656_3124841/checkpoints/checkpoint-022309`.
The prompt plan comes from that run's `evaluation/prepared/plan.json`.
Override `DRAFT`, `MODEL`, `RUN`, `BENCHMARK_PLAN`, `VLLM_SOURCE_ROOT` or
`CANN_ROOT` if necessary. `BENCHMARKS=gsm8k,math500` selects a suite subset.
No dataset download is attempted. Smaller cached benchmarks retain their
available count (for example AIME25: 30, MT-Bench: 80).

Baseline runs are **off by default**. To validate exact greedy token agreement,
use `WITH_BASELINE=1 ... smoke`, or pass an existing compatible AR
`BASELINE=/absolute/path/per_prompt.jsonl` to `smoke`/`evaluate`. Prompt token
IDs, target and output budget must match. A mismatch saves provisional results
then raises an error; no correctness guard is disabled. A run without a baseline
reports acceptance/throughput but no new speedup or parity claim.

## Results and profiling

New files are saved only below this separate checkout:

```text
four-anchor-bonus20-inference/output/bonus_addon_<timestamp>_<pid>/
  two_pass/{summary.json,per_prompt.jsonl,eval.log}   # smoke/evaluate
  summary.md                                      # benchmarks action
  summary.csv
  suite_summary.json
  plan.json
  gsm8k/two_pass/{summary.json,per_prompt.jsonl,eval.log}
  ...other benchmarks...
```

Per-prompt JSON records native acceptance counters; `summary.json` records
both query lengths, 15 speculative tokens, the 16-token verification block,
four first-pass drafter states in round two, and measured extra target calls.
Training outputs, queues and epoch-evaluation tables are not overwritten.

The low-level launcher accepts `PROFILE_STEPS` for synchronized component
timings and `DRAFT_GRAPH=1` for optional drafter ACLGraph testing. The supplied
benchmark launcher deliberately keeps both off for comparison. NPU graph
execution has not been validated for this patch. No performance result is
included until the new policy runs on the Ascend machine.

## Validation and push

CPU tests exercise all 15 restart boundaries, short-prefix padding, explicit
bonus conditioning, all four drafter-state sources, dense/cached numerical
agreement, rejected-cache isolation, tiny-Qwen greedy parity with one target
forward per cycle, output budgets, native counter guards, frozen benchmark
reuse and installation isolation. These are not Ascend kernel tests.

To review and push this separate branch after installation:

```bash
cd /home/n84449292/a84384644/attention_head_forked/four-anchor-bonus20-inference
git diff --stat
python four_anchor/scripts/install_bonus_addon.py --stage
git diff --cached --check
git commit -m "Add bonus embedding and four drafter-state anchors for redrafting"
git push -u origin four-anchor-bonus20-inference
```

The `--stage` helper stages only the explicit inference-package allowlist; it
does not add outputs, checkpoints or training changes. To include benchmark
results later, copy just the small summary files into `four_anchor/results/`
and stage those explicitly.
