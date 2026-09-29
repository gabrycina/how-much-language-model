"""EXPLORATORY (not pre-registered): Jev vs each 7B model pooled over T15 and T12 test sentences.

Settings are chosen per participant on its own development split, exactly as in prereg_analysis.py;
the bootstrap resamples sentences within each participant (stratified) and pools word errors.
"""
import json, pickle
import numpy as np, jiwer
import analyze as A
from revision_analysis import split
from prereg_analysis import LAMBDAS, ALPHAS

E = {}
def err(r, h):
    if (r, h) not in E:
        m = jiwer.process_words([r], [h]); E[(r, h)] = m.substitutions + m.deletions + m.insertions
    return E[(r, h)]

def participant(lists, sdir, prefix):
    cache, _ = pickle.load(open(lists, "rb")); dev, test = split(cache)
    zero = {s: np.zeros(len(nb)) for s, nb in cache.items()}
    wer = lambda p, ks: 100 * sum(err(s, p[s]) for s in ks) / sum(len(s.split()) for s in ks)
    picks = lambda sc, al, lam: A.picks(cache, sc, al, acoustic_scale=lam)
    lamA = min(LAMBDAS, key=lambda l: wer(picks(zero, 0.0, l), dev))
    out = {"len": np.array([len(s.split()) for s in test], float)}
    for arm in ("jev", "opt", "qwen"):
        raw = pickle.load(open(f"{sdir}/{prefix}{arm}.pkl", "rb"))["scores"]; sc = {s: raw.get(s, zero[s]) for s in cache}
        alA = min(ALPHAS, key=lambda x: wer(picks(sc, x, lamA), dev))
        _, lamB, alB = min((wer(picks(sc, x, l), dev), l, x) for l in LAMBDAS for x in ALPHAS)
        for prot, (al, lam) in {"A": (alA, lamA), "B": (alB, lamB)}.items():
            p = picks(sc, al, lam); out[(arm, prot)] = np.array([err(s, p[s]) for s in test], float)
    return out

P = [participant("../data/lists_published.pkl", "../data/scores_published", "kaldi_"),
     participant("../data/lists_t12.pkl", "../data/scores_t12", "t12_")]
rng = np.random.default_rng(0); res = {}
idx = [[rng.integers(0, len(p["len"]), len(p["len"])) for _ in range(5000)] for p in P]
for prot in ("A", "B"):
    for m in ("opt", "qwen"):
        num = sum(p[("jev", prot)].sum() - p[(m, prot)].sum() for p in P); den = sum(p["len"].sum() for p in P)
        boots = []
        for b in range(5000):
            n = sum(P[k][("jev", prot)][idx[k][b]].sum() - P[k][(m, prot)][idx[k][b]].sum() for k in range(2))
            d = sum(P[k]["len"][idx[k][b]].sum() for k in range(2)); boots.append(n / d * 100)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[f"jev-{m}/{prot}"] = {"diff": num / den * 100, "ci": [lo, hi]}
        print(f"pooled {prot} jev-{m:5s} {num/den*100:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
json.dump(res, open("../analysis/pooled_exploratory.json", "w"), indent=1, default=float)
