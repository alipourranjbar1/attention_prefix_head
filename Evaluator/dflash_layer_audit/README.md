# Qwen3 / DFlash layer correspondence

Run from the active PyTorch / CANN shell on server 133. Choose one allocated NPU, then run `bash Evaluator/dflash_layer_audit/run_133.sh --fit 64 --holdout 32`. Set `TARGET`, `DRAFT`, `DATASET`, or `OUT` to override local paths. The launcher can retrieve the official DFlash checkpoint on a cache miss. The checkpoint supplies executable Python model code. Use a trusted local copy when overriding DRAFT. The target needs its Hugging Face tokenizer and weights. The experiment uses PyTorch and Transformers eager attention. It requires no vLLM worker or Ascend operator compilation.

## What one observation means

One distinct, tokenized prefix produces one greedy proposal block. Target prefill produces the anchor and the drafter's prescribed conditioning features. The drafter receives that anchor and 15 mask tokens. A second causal target forward verifies the proposed trajectory. At slot s (1..15) the drafter residual at s is paired with the target residual *before* the proposed token at s. Both hooks capture decoder block outputs before the model's final RMSNorm. The target never supplies a future state to the drafter. All slots including those after the first rejection are scored, even if a generation would stop early at EOS. This measures the first proposal at the sampled context, rather than a whole decode trajectory.

The input may be a local Hugging Face saved dataset or JSONL. Rows can contain input_ids, a string prompt, or text. On Qwen chat-tokenized rows the first assistant header ends the prompt. Without that header, the sampler uses the first prefix-tokens IDs. Text rows use the end of the supplied text. Use already formatted prompts for text rows. A fixed seed visits rows without replacement. Duplicate effective token prefixes are excluded before splitting. Prompt hashes and source row indices are recorded. For a final estimate, reserve another untouched set after using the audit to choose layers.

## Reading the output

* `scores.csv` gives every layer pair at every slot on held-out prefixes. Layer indices start at zero. The official 5-by-36 model yields 2,700 rows.
* `best_layers.csv` gives the target winner per metric and the score for the target's penultimate block. This is block 34 for a 36-block target, the input boundary for block 35. The configured target conditioning IDs [1,9,17,25,33] have a different role.
* The six SVG heatmaps give the mean of per-slot scores. Open one in a browser and hover a cell for its value. Each figure has its own color range.
* Cosine is averaged over prompts. Relative MSE is total squared vector error divided by total target energy. Linear CKA uses centered Gram matrices. The debiased CKA variant uses U-centered Gram matrices and may be negative; ordinary CKA can look high in small samples.
* Each layer pair has a separate affine ridge probe for each slot. The intercept is unpenalized. Its ridge penalty equals `--ridge` times mean diagonal of the centered training Gram matrix. Probe R² compares held-out squared error to prediction by the training target mean for that slot. R² near zero means little benefit over that baseline. Higher cosine, CKA and R² are better. Lower raw and probe relative MSE are better.
* `prompts.csv` includes greedy accepted-prefix length (0..15) for each sample; `run.json` includes configuration and activation shapes.

The 64/32 run is a pilot. With 64 calibration examples per slot a centered probe can learn at most 63 input directions. Its score tests a position-specific map. A single adapter shared across positions is a separate experiment. Host RAM for the two activation arrays is about 0.60 GB for 96 prompts at full model size and scales linearly with prompt count. Activations are released when the process exits. No checkpoint weights are updated.
