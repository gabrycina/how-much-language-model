"""Re-tune the decoder's acoustic weight jointly with the interpolation weight alpha.

The main tables hold the acoustic weight at the published 0.3. Here both weights are
chosen per arm on the development split only, the test split is scored once, and each arm
is compared against no rescoring with a one-sided paired bootstrap (5,000 resamples).
Holm correction runs over the seven arms of Table 2. Produces Table 3 of the paper.

    python src/joint_tuning.py          # ~10 min on a laptop CPU
"""
import json, os, pickle
import numpy as np, jiwer
import analyze as A
from revision_analysis import split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SCALES = [0, .1, .2, .25, .3, .35, .4, .45, .5, .6, .8, 1.0, 1.25, 1.5, 2, 3, 5]
ALPHAS = np.round(np.arange(0, 1.001, .05), 2)
ARMS = ["distilgpt2", "gpt2small", "gpt2medium", "gpt2", "opt", "qwen", "jev",
        "laya", "optchoice", "qwenchoice"]
HOLM_FAMILY = ["distilgpt2", "gpt2small", "gpt2medium", "gpt2", "opt", "qwen", "jev"]

cache, _ = pickle.load(open(os.path.join(DATA, "lists_published.pkl"), "rb"))
dev, test = split(cache)
_err = {}

def errs(ref, hyp):
    if (ref, hyp) not in _err:
        m = jiwer.process_words([ref], [hyp])
        _err[(ref, hyp)] = m.substitutions + m.deletions + m.insertions
    return _err[(ref, hyp)]

def wer(p, keys):
    return 100 * sum(errs(s, p[s]) for s in keys) / sum(len(s.split()) for s in keys)

def picks(scores, alpha, scale):
    return A.picks(cache, scores, alpha, acoustic_scale=scale)

zero = {s: np.zeros(len(nb)) for s, nb in cache.items()}
fp_scale = min(SCALES, key=lambda sc: wer(picks(zero, 0.0, sc), dev))
fp = picks(zero, 0.0, fp_scale)
ln = np.array([len(s.split()) for s in test], float)
eb = np.array([errs(s, fp[s]) for s in test], float)
ix = np.random.default_rng(0).integers(0, len(test), (5000, len(test)))

out = {"no_rescoring": {"scale": fp_scale, "test_wer": wer(fp, test)}}
for arm in ARMS:
    raw = pickle.load(open(os.path.join(DATA, "scores_published", f"kaldi_{arm}.pkl"), "rb"))["scores"]
    sc_ = {s: raw.get(s, zero[s]) for s in cache}   # unscored lists fall back to the first pass
    dw, scale, alpha = min((wer(picks(sc_, a, sc), dev), sc, float(a)) for sc in SCALES for a in ALPHAS)
    p = picks(sc_, alpha, scale)
    ea = np.array([errs(s, p[s]) for s in test], float)
    d = (ea[ix].sum(1) - eb[ix].sum(1)) / ln[ix].sum(1) * 100
    out[arm] = {"scale": scale, "alpha": alpha, "test_wer": wer(p, test),
                "delta": wer(p, test) - wer(fp, test), "ci": list(np.percentile(d, [2.5, 97.5])),
                "p": float(np.mean(d >= 0))}
    print(arm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[arm].items()}, flush=True)

ps = sorted((out[a]["p"], a) for a in HOLM_FAMILY); m = len(ps); run = 0.0
for i, (p, a) in enumerate(ps):
    run = max(run, min(1.0, (m - i) * p)); out[a]["p_holm"] = run
print("Holm:", {a: round(out[a]["p_holm"], 3) for a in HOLM_FAMILY})
json.dump(out, open(os.path.join(HERE, "..", "analysis", "joint_tuning.json"), "w"), indent=1, default=float)
