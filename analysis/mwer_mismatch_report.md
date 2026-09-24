# MWER pipeline mismatch — root cause and fix (2026-09-22)

## Symptom
`round3_mwer_train.py`'s startup self-check printed:
- cached GPT-2-small @ α=0.4, test WER **8.968** (expected **7.983**)
- no-rescoring test WER **9.389** (expected **8.091**)

The run then aborted before training, per its own protocol.

## Root cause: wrong dev/test split (insertion order vs canonical shuffled-sorted split)

The canonical round-3 pipeline lives on the Nebius L40S box at
`~/b2t/work/make_results.py` (the script that produced `revision_results.json`,
`results_kaldi.json`, `SUMMARY.md`, and the paper's Table 1). Its split:

```python
# ~/b2t/work/make_results.py, lines 19-24
def split(cache, seed=0):
    keys = sorted(cache)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(keys))
    k = int(len(keys) * DEV_FRAC)
    return [keys[i] for i in idx[:k]], [keys[i] for i in idx[k:]]
```

i.e. **sorted** keys, permuted with **`np.random.default_rng(0)`** (the new
Generator API — not `RandomState`, not `random.shuffle`), first 30% = dev (419),
rest = test (978).

`round3_mwer_train.py::load()` instead used **pickle insertion order** with no
shuffle:

```python
keys = list(lists_dict.keys())                      # insertion order
n_dev = int(len(keys) * 0.30)
dev_keys, test_keys = keys[:n_dev], keys[n_dev:]
```

That assigns a *different* 978 sentences to test. Everything else in the
self-check was already correct — this was proven by reproducing both canonical
numbers **exactly** (to all 16 decimal places) with the MWER recipe on the
canonical split:

| check | insertion-order split | canonical split | paper value |
|---|---|---|---|
| cached gpt2-small @ α=0.4, test | 8.968 | 7.983450812136071 | 7.983 |
| no-rescoring, test | 9.389 | 8.090714066809685 | 8.091 |

## Ruled out (verified identical between the two pipelines)
- **WER computation / normalisation**: same word-level Levenshtein on raw
  lowercased, punctuation-free strings; candidate texts carry no case/punct.
- **Interpolation recipe**: both use `s(h) = (1−α)·z(decoder) + α·z(LM)` with
  per-list z-normalisation (population std) of both terms — matches the paper's
  §"Score combination" and the V0 arm definition.
- **Decoder score**: `0.3·acoustic + ngram` in both (acoustic_scale 0.3 in
  `lists_kaldi.pkl` meta); lists are pre-sorted by decoder score, so 1-best =
  argmax in both.
- **N-best list size**: full lists in both (main 100-best regime; mean list
  length 38.7).
- **Alpha grid / tie-breaking**: canonical `tune_alpha` iterates 0.00→1.00 step
  0.05 keeping strictly-better (ties → smallest α); the MWER script's
  `min(...)` has identical tie behaviour.

## The fix (applied)
`analysis_out/round3_mwer_train.py`, `load()` now implements the canonical split;
module docstring corrected (it previously claimed "insertion order"):

```python
keys = sorted(lists_dict.keys())
idx = np.random.default_rng(0).permutation(len(keys))
n_dev = int(len(keys) * 0.30)                        # 419
dev_keys = [keys[i] for i in idx[:n_dev]]
test_keys = [keys[i] for i in idx[n_dev:]]
train_keys, held_keys = dev_keys[:300], dev_keys[300:]
```

## Verification (script's own code, no training run)
- `load()` → train 300 / heldout 119 / dev 419 / test 978; train∪heldout ∩ test = ∅
- check 1: cached gpt2-small @ α=0.4 test WER **7.983** ✓
- check 2: no-rescoring test WER **8.091** ✓

## Notes for the GPU run
- No other change to the script was needed; the training protocol
  (top-20 MWER, AdamW 1e-5, batch 4 lists, 3 epochs, early stop on heldout,
  α grid 0.00–1.00 on full dev, paired bootstraps) is untouched.
- `train_keys`/`held_keys` are now the first 300 / last 119 of the *canonical*
  dev ordering, matching the protocol spec.
- Do not reintroduce an insertion-order split anywhere else (e.g. if the eval
  harness on the GPU box re-splits, it must use the same `sorted` +
  `default_rng(0)` recipe).
