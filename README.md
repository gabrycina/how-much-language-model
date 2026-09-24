# How Much Language Model Does a Speech Neuroprosthesis Need?

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

- **The stage contributes a fraction of a word-error point.** Sweeping 82M to 7.6B
  parameters moves WER from 8.1% to 7.8%. Every arm's 95% interval overlaps every other's.
  After Holm correction over seven arms, no open model separates from no rescoring at all.
- **The gain depends on first-pass strength.** Varying only the decoder's acoustic scale on
  fixed candidate sets, rescoring is worth 6.8 points when the decoder errs at 19.7% and
  0.3 points at 8.1%.
- **The single-pass shortcut fails.** Asking a model to name the best candidate in one
  forward pass, rather than scoring all 100, fails for every model tested. They answer from
  where a label sits in the list rather than from the candidate beside it.
- **It is not an artefact of score combination.** Four interpolation schemes, a jointly fit
  multi-arm reranker, and MWER fine-tuning of the smallest rescorer all land within noise.

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
