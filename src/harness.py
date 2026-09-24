"""Evaluation harness: build n-best lists, rescore them, measure WER and latency.

Scorers are interchangeable. Each takes the list of hypotheses and returns one score per
hypothesis, exactly like `rescore_with_gpt2` in the Brain-to-Text baseline, and the
combination is the baseline's own:

    total = acoustic_scale*acoustic + (1-alpha)*ngram + alpha*rescorer
"""

import csv
import time

import jiwer
import numpy as np

from nbest import NBestBuilder

ACOUSTIC_SCALE = 0.35
ALPHA = 0.55


def load_sentences(path="sentences.csv", limit=None):
    rows = list(csv.DictReader(open(path)))
    s = [r["text"].strip() for r in rows if r["text"].strip()]
    return s[:limit] if limit else s


def pick(nbest, new_scores=None, alpha=ALPHA):
    hyps = [h for h, _, _ in nbest]
    ac = np.array([a for _, a, _ in nbest])
    lm = np.array([l for _, _, l in nbest])
    lm = (lm - lm.mean()) / (lm.std() + 1e-9)
    if new_scores is None:
        total = ACOUSTIC_SCALE * ac + lm
    else:
        ns = np.asarray(new_scores, dtype=float)
        ns = (ns - ns.mean()) / (ns.std() + 1e-9)
        total = ACOUSTIC_SCALE * ac + (1 - alpha) * lm + alpha * ns
    return hyps[int(np.argmax(total))]


def evaluate(sentences, builder, scorer=None, n=100, sub_rate=0.30, verbose=False):
    """Returns (WER, mean latency ms, oracle WER, extra diagnostics)."""
    refs, hyps, lats, oracle, top1 = [], [], [], [], []
    for s in sentences:
        nb = builder.build(s, n=n, sub_rate=sub_rate)
        t0 = time.perf_counter()
        scores = scorer(s, [h for h, _, _ in nb]) if scorer else None
        lats.append((time.perf_counter() - t0) * 1000)
        hyps.append(pick(nb, scores))
        refs.append(s)
        oracle.append(min(jiwer.wer(s, h) for h, _, _ in nb))
        top1.append(s == nb[int(np.argmax([a for _, a, _ in nb]))][0])
    return {
        "wer": float(jiwer.wer(refs, hyps)),
        "latency_ms": float(np.mean(lats)),
        "oracle_wer": float(np.mean(oracle)),
        "truth_in_list": 100.0,
        "acoustic_top1_correct": float(np.mean(top1) * 100),
        "n_sentences": len(sentences),
    }
