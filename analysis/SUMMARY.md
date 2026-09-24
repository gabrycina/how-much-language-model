# Revision analysis summary (cached-score runs, 21 Sep 2026)

Split: dev 419 / test 978 sentences, seed 0, first 30% = dev. All hyperparameters tuned on dev only.
Sanity check reproduced: first-pass (none) 8.091%, OPT V0 7.800%, Jev V0 7.462%.

## A. Score-combination ablations
Per-list z-normalised interpolation (V0), raw likelihood + learned global scale (V1),
per-arm temperature with no extra weight (V2), and a discriminative linear MWER-style
interpolation (V3) are all within noise of each other on every arm. No variant beats V0
significantly (all paired one-sided p >= 0.16 except one distilgpt2-V3 comparison at p=0.019,
which reverses the pattern of the other arms and was not replicated elsewhere).
- OPT-125M: V0: 7.8, V1: 7.708, V2: 7.708, V3: 7.83 (dev hyperparams: V0 alpha=0.3, V1 beta=2.89, V2 T=0.344, V3 w=[0.5, 0.25])
- Qwen2.5-0.5B: V0: 7.754, V1: 7.769, V2: 7.6, V3: 7.769 (dev hyperparams: V0 alpha=0.45, V1 beta=1.43, V2 T=0.467, V3 w=[1.5, 1.25])
- GPT-2: V0: 7.846, V1: 8.091, V2: 8.106, V3: 7.876 (dev hyperparams: V0 alpha=0.25, V1 beta=1.43, V2 T=0.517, V3 w=[3.0, 1.75])
- GPT-2-medium: V0: 7.922, V1: 8.029, V2: 8.091, V3: 7.892 (dev hyperparams: V0 alpha=0.2, V1 beta=0.702, V2 T=0.517, V3 w=[2.5, 0.75])
- GPT-2-small: V0: 7.983, V1: 8.121, V2: 8.367, V3: 8.029 (dev hyperparams: V0 alpha=0.4, V1 beta=0.492, V2 T=0.517, V3 w=[1.2500000000000004, 0.8000000000000002])
- DistilGPT-2: V0: 8.106, V1: 8.06, V2: 8.504, V3: 8.029 (dev hyperparams: V0 alpha=0.2, V1 beta=0.492, V2 T=0.633, V3 w=[1.1, 0.25])

## B. Multi-arm discriminative baseline
A linear MWER-style combination of first-pass + 5 LM arms (weights {'first': 1.0, 'opt': 0.40000000000000036, 'qwen': 0.40000000000000036, 'gpt2': -0.19999999999999973, 'gpt2small': 0.0, 'distilgpt2': 0.0})
reaches 7.677% test WER vs 7.8% for the best single
arm (OPT V0), paired p=0.161 — a small, non-significant gain. Even a
jointly tuned discriminative combination barely moves the needle.

## C. Controlled first-pass sweep
Fixing the decoder and the candidate lists, varying only the acoustic scale:
- scale 0: first-pass 19.721%, OPT (alpha=1.0) 12.964%, gain 6.758pp
- scale 0.05: first-pass 17.315%, OPT (alpha=0.8) 13.178%, gain 4.137pp
- scale 0.1: first-pass 14.036%, OPT (alpha=0.65) 12.412%, gain 1.624pp
- scale 0.15: first-pass 11.477%, OPT (alpha=0.5) 10.512%, gain 0.965pp
- scale 0.2: first-pass 9.301%, OPT (alpha=0.35000000000000003) 8.704%, gain 0.598pp
- scale 0.3: first-pass 8.091%, OPT (alpha=0.30000000000000004) 7.8%, gain 0.291pp
- scale 0.45: first-pass 8.397%, OPT (alpha=0.45) 7.248%, gain 1.149pp
- scale 0.6: first-pass 9.454%, OPT (alpha=0.5) 7.738%, gain 1.716pp
- scale 0.8: first-pass 10.19%, OPT (alpha=0.5) 7.861%, gain 2.329pp
- scale 1.0: first-pass 10.665%, OPT (alpha=0.5) 8.014%, gain 2.651pp
Oracle (truth always picked) test WER: 2.482% — constant across scales,
so the lists themselves contain the headroom. Rescoring gain is largest when the first pass
is weak (6.8pp at 19.7% WER) and shrinks to ~0.3pp at the 8.1% operating point; it grows again
when the scale is raised past 0.3, which just degrades the first pass.

## D. Truth in list
Test: 874 sentences with truth in list, 104 without.
- none: in-list WER 4.367% (delta 0.0, p=None); out-of-list WER 36.556% (delta 0.0, p=None)
- opt: in-list WER 4.055% (delta -0.312, p=0.0352); out-of-list WER 36.424% (delta -0.132, p=0.433)
- qwen: in-list WER 4.037% (delta -0.329, p=0.0774); out-of-list WER 36.159% (delta -0.397, p=0.3418)
- gpt2small: in-list WER 4.28% (delta -0.087, p=0.3456); out-of-list WER 36.291% (delta -0.265, p=0.3602)
- jev: in-list WER 3.795% (delta -0.572, p=0.0148); out-of-list WER 35.497% (delta -1.06, p=0.1228)
Abstention at 50% coverage (sentence error): jev in-list 6.6%, out-of-list 100%;
opt (temperature-calibrated) in-list 1.8%, out-of-list 100%. When the truth is absent,
both systems fail on every kept sentence — abstention cannot rescue out-of-list sentences.

## E. Error analysis
- none: 7.524 subs, 0.215 ins, 0.352 dels per 100 ref words
- opt: 7.202 subs, 0.291 ins, 0.306 dels per 100 ref words
- jev: 6.635 subs, 0.414 ins, 0.414 dels per 100 ref words
Errors are dominated by substitutions. On the 94 test sentences where OPT overrides the
decoder top-1, the correction rate is 26% / 16% / 53% across low/mid/high decoder-margin
terciles (point-biserial r=0.12): overrides of a confident decoder are the most likely to
be right, but the relationship is weak and non-monotonic — margin does not reliably
predict an override's success.

## F. Typed label prior
- optchoice: used 13/100 labels, entropy 2.422 bits vs 6.644 uniform; top-5 labels take 91.1% of selections; label 0 alone takes 34.8%.
- qwenchoice: used 31/100 labels, entropy 3.423 bits vs 6.644 uniform; top-5 labels take 71.8% of selections; label 0 alone takes 29.0%.
Typed selection collapses onto a handful of labels — strong evidence the prompt/label design,
not the model, is the bottleneck.
