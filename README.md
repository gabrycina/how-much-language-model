# Language-model rescoring for a speech neuroprosthesis

Code, cached data and two papers on the second-pass rescoring stage of a speech
brain-computer interface, on real intracortical recordings from participant T15.

**Paper (v2)** ([`paper-v2/`](paper-v2/paper.pdf)): *Jev versus 7B Language Models for
Speech-Neuroprosthesis Rescoring: A Pre-Registered Comparison in Two Participants.* One call to Jev, a
hosted typed-decision model, rescores the candidate list about as well as OPT-6.7b or Qwen2.5-7B
scoring every candidate. On T15 Jev is ahead in all four comparisons (two models, two tuning
protocols) and at most 0.20 points of word error behind at the 95% bound. A replication on T12,
pre-registered in [`PREREGISTRATION_T12.md`](PREREGISTRATION_T12.md) before any T12 data was used,
meets the margin in two of four comparisons; the other two are inconclusive (point estimates +0.10
and +0.08). Open models asked the same way fail because they answer from label position. Jev costs
$0.07-0.12 per thousand sentences; its latency is network-bound (62 ms of server time) and not below a
local 7B model. Reproduce with `src/prereg_analysis.py`, `src/collapse_diag.py` and
`src/pooled_exploratory.py` (cached data, CPU only).

The earlier single-participant version is in [`paper-short/`](paper-short/typed_rescoring.pdf). It
reports 83 test sentences overlapping the phoneme network's training text; the correct count is 107
(the earlier check normalised apostrophes on one side only). No conclusion changes.

**Extended report** ([`paper/`](paper/rescoring.pdf)): *How Much Language Model Does a
Speech Neuroprosthesis Need?* The scale study, first-pass-strength sweep and failure
analysis behind the short paper.

## Extended report

Code, cached data and paper for a study of the second-pass language-model rescoring stage
in a speech brain-computer interface.

A speech neuroprosthesis decodes attempted speech from intracortical activity, and every
high-performance system ends by rescoring the decoder's candidate sentences with a language
model. That stage is the only part of the pipeline needing a multi-billion-parameter model,
so it sets the hardware a patient has to carry. The published evidence for it comes from
systems whose first-pass decoder was weak. This study asks whether it still pays once the
decoder improves.

## Findings

Holding the published T15 pipeline fixed (its Kaldi/WFST decoder and OpenWebText 3-gram)
and scoring one cached 100-candidate list per sentence with every rescorer:

- **Once the decoder is strong, the stage is worth under a point.** At the published
  acoustic weight, sweeping 82M to 7.6B parameters moves WER from 8.1% to 7.8%, and after
  Holm correction no open model separates from no rescoring.
- **Re-tuning the decoder's weight lifts the 7B models.** Tuning the acoustic weight
  jointly with the rescorer (dev split only) takes OPT-6.7b to 7.2% and Qwen2.5-7B to 7.4%,
  both significant after Holm correction; nothing under 1B parameters helps under either
  protocol (`src/joint_tuning.py`).
- **The gain depends on first-pass strength.** Varying only the decoder's acoustic scale on
  fixed candidate sets, rescoring is worth 6.8 points when the decoder errs at 19.7% and
  0.3 to 0.9 points at 8.1%.
- **The single-pass shortcut fails for general language models.** Asking a model to name
  the best candidate in one forward pass, rather than scoring all 100, fails for every open
  model tested. They answer from where a label sits in the list rather than from the
  candidate beside it. The one purpose-built typed model tested does not collapse.
- **The interpolation rule is not the bottleneck.** Four interpolation schemes, a jointly
  fit multi-arm reranker, and MWER fine-tuning of the smallest rescorer all land within
  noise. The decoder's own acoustic weight matters more than any of them.

Every arm is untuned and text-only, so these are lower bounds on what the stage could
deliver with training or acoustic access.

## Layout

```
paper/      LaTeX source, generated tables, and the compiled PDF
src/        pipeline: RNN inference, decoding, rescoring arms, analysis
data/       cached n-best lists and per-arm scores (no model re-runs needed)
analysis/   result JSONs, statistics, and notes behind each number
```

`data/lists_published.pkl` holds the 1,397 candidate lists from the published decoder;
`data/lists_weak.pkl` holds the earlier, weaker first pass kept as a second data point.
Per-arm scores in `data/scores_*` mean every table can be recomputed without a GPU.

## Reproducing

Recompute all tables and statistics from the cached scores:

```bash
python src/make_results.py          # main results table and significance
python src/revision_analysis.py     # combination ablations, abstention, token accounting
```

Regenerating the lists themselves needs the T15 recordings from the Brain-to-Text
benchmark, the pretrained RNN, and the published Kaldi decoder with its OpenWebText
3-gram (~40 GB, ~60 GB RAM). `src/extract_logits.py` and `src/build_lists_kaldi.py` cover
that path.

## Data

Neural recordings and the pretrained RNN come from the Brain-to-Text benchmark
(https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text). The language model is the
OpenWebText 3-gram released with Willett et al. (2023). Neither is redistributed here.

## Licence

Paper and text: CC BY 4.0. Code: MIT. See `LICENSE`.
