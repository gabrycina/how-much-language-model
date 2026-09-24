"""Decode n-best lists once per regime and cache them.

Every rescoring arm scores these exact lists, so differences between arms are the
rescorer and nothing else. Two regimes are built: the easy one used in the first draft,
and a hard one where the beam is narrow enough that the true sentence often does not
make the list at all - which is where a reviewer would expect calibration to degrade.
"""
import argparse
import pickle
import time

import jiwer
import numpy as np

from decoder import BeamDecoder
from harness import load_sentences
from nbest import NBestBuilder

REGIMES = {
    # name:   phoneme error, beam, n-best, extra dictionary words
    "easy": dict(er=0.25, beam=400, n=60, extra=0),
    "hard": dict(er=0.35, beam=400, n=60, extra=12000),
}


def build(regime, seed, n_sentences, out):
    cfg = REGIMES[regime]
    er, beam, n = cfg["er"], cfg["beam"], cfg["n"]
    sents = load_sentences()
    train, test = sents[:1100], sents[1100:1100 + n_sentences]
    builder = NBestBuilder(lm_sentences=train, lexicon_sentences=sents, seed=seed,
                           extra_words=cfg["extra"])
    dec = BeamDecoder(builder, beam=beam)

    t0 = time.perf_counter()
    cache, oracle, inlist = {}, [], []
    for s in test:
        nb = dec.decode(builder.observe(s.split(), er), n=n)
        if not nb:
            continue
        cache[s] = nb
        oracle.append(min(jiwer.wer(s, h) for h, _, _ in nb))
        inlist.append(s in [h for h, _, _ in nb])
    meta = {"regime": regime, "seed": seed, "error_rate": er, "beam": beam, "nbest": n,
            "extra_words": cfg["extra"],
            "n": len(cache), "oracle_wer": float(np.mean(oracle)) * 100,
            "truth_in_nbest": float(np.mean(inlist)) * 100,
            "mean_list_len": float(np.mean([len(v) for v in cache.values()])),
            "lexicon": len(builder.vocab), "lm_train": len(train)}
    pickle.dump((cache, meta), open(out, "wb"))
    print(f"{regime} seed={seed}: n={meta['n']}  oracle {meta['oracle_wer']:.1f}%  "
          f"truth-in-list {meta['truth_in_nbest']:.0f}%  "
          f"mean |list| {meta['mean_list_len']:.0f}  ({time.perf_counter()-t0:.0f}s)")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="easy", choices=list(REGIMES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sentences", type=int, default=326)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    build(a.regime, a.seed, a.sentences,
          a.out or f"lists_{a.regime}_s{a.seed}.pkl")
