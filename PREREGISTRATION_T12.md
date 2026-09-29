# Pre-registered confirmatory analysis: participant T12

Written and committed **before any T12 data was downloaded, decoded or scored** by us.
The git commit that adds this file is its timestamp. Any deviation will be reported as such.

## Why
On T15 the short paper claims that Jev (a hosted typed-decision model) *matches* two 7B rescorers,
using a margin (0.2 points of word error) chosen after seeing the T15 results. This analysis tests
the same claim on a second participant with the margin fixed in advance.

## Data
Willett et al. 2023, "A high-performance speech neuroprosthesis", Dryad doi:10.5061/dryad.x69p8czpq
(CC0), `competitionData`: partitions `train` and `test` (transcripts available). The
`competitionHoldOut` partition (no transcripts) is not used.

## Phoneme decoder
- Architecture and hyperparameters: the public baseline (github.com/cffan/neural_seq_decoder,
  `scripts/train_model.py`): 5-layer bidirectional GRU, 1024 units, 10,000 batches of 64, seed 0,
  tx1 + spike power of area 6v (256 features), block-wise z-scoring.
- Trained on the `train` partition only. Checkpoint selection uses a validation set of 10% of `train`
  trials per session (seed 0). **The `test` partition is never used for training or model selection.**
  (The baseline's own trainer selects the checkpoint on `test`; we do not use that behaviour.)

## Candidate lists
Same Kaldi/WFST decoder, OpenWebText 3-gram, lexicon and decoder options as the T15 paper, 100-best
lists. Transcripts are normalised exactly as for T15 (lowercase, letters, apostrophes, spaces).

## Split
`test`-partition sentences are split 30% development / 70% test with the paper's `split()` function
(sorted keys, `numpy.random.default_rng(0)` permutation).

## Arms
No rescoring; OPT-6.7b and Qwen2.5-7B (per-candidate log-likelihood, same code as T15); Jev (same
request as T15, model version recorded at run time). Controls: Laya, OPT-6.7b and Qwen2.5-7B given
the typed prompt. Exploratory only: an open 0.6B typed-decision model, if it runs.

## Protocols
- **A (fixed decoder weight):** the acoustic weight λ is chosen on the development split for the
  first pass alone (grid: the 17 values of the T15 re-tuning grid); α is tuned per arm on development
  (21 values in [0, 1]).
- **B (re-tuned):** λ and α are tuned jointly per arm on development (17 × 21 grid).
The test split is scored once per arm and protocol.

## Primary hypothesis and decision rule
For each 7B model M in {OPT-6.7b, Qwen2.5-7B} and each protocol in {A, B}, compute the paired
difference WER(Jev) − WER(M) on the test split with a percentile bootstrap over sentences
(5,000 resamples, `default_rng(0)`), two-sided 95% interval.

**Margin:** δ = 0.025 × (first-pass WER on the development split). This is the T15 margin
(0.2 points at 8.09% first-pass WER) expressed relative to first-pass error, because T12's first pass
is expected to be weaker. The absolute 0.2-point criterion is reported alongside.

**Jev matches M** under a protocol if the upper bound of that interval is below δ.
The claim is **confirmed** if this holds in all four comparisons, **partly confirmed** if in some,
and **not confirmed** otherwise. All four are reported regardless.

## Secondary analyses
- Each arm against no rescoring: one-sided paired bootstrap, Holm correction over the three arms
  within each protocol.
- Test sentences whose text also occurs in the `train` partition: counted, and the primary analysis
  repeated without them.
- Cost (billed tokens) and latency, measured as for T15.
