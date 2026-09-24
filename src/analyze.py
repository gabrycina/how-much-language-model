"""Analysis over cached scores: interpolation sweeps, bootstrap intervals, calibration.

None of this touches a model. Every arm's raw per-candidate scores were saved once by
runscores.py, so sweeping alpha or resampling for confidence intervals is free.
"""
import pickle

import jiwer
import numpy as np

ACOUSTIC_SCALE = 0.35
ALPHA = 0.55


def _z(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-9)


def picks(cache, scores, alpha=ALPHA, acoustic_scale=ACOUSTIC_SCALE):
    """The hypothesis each arm would speak, for every sentence it scored.

    The first-pass score is the published combination of the decoder's own acoustic and
    n-gram terms, `acoustic_scale * ac + lm`, kept in its native scale. The rescorer is
    then blended against it after z-normalising both, because rescorer scores live on
    scales that differ by orders of magnitude between arms (a sentence log-likelihood
    versus a log-probability over a 100-way menu) and a raw interpolation weight would
    not be comparable across them.
    """
    out = {}
    for s, nb in cache.items():
        hyps = [h for h, _, _ in nb]
        ac = np.array([a for _, a, _ in nb], dtype=float)
        lm = np.array([l for _, _, l in nb], dtype=float)
        first = acoustic_scale * ac + lm
        if scores is None or s not in scores:
            total = first
        else:
            total = (1 - alpha) * _z(first) + alpha * _z(scores[s])
        out[s] = hyps[int(np.argmax(total))]
    return out


def metrics(cache, scores, alpha=ALPHA, subset=None):
    p = picks(cache, scores, alpha)
    keys = [s for s in (subset or p) if s in p]
    refs = keys
    hyps = [p[s] for s in keys]
    return {"wer": float(jiwer.wer(refs, hyps)) * 100,
            "sent_acc": float(np.mean([r == h for r, h in zip(refs, hyps)])) * 100,
            "n": len(keys)}


def bootstrap_wer(cache, scores, alpha=ALPHA, B=2000, seed=0):
    """Percentile interval over sentence resamples - the sentences are the sample unit."""
    p = picks(cache, scores, alpha)
    keys = list(p)
    rng = np.random.default_rng(seed)
    # jiwer over a resample is slow; accumulate per-sentence edit counts instead.
    errs, lens = [], []
    for s in keys:
        m = jiwer.process_words([s], [p[s]])
        errs.append(m.substitutions + m.deletions + m.insertions)
        lens.append(len(s.split()))
    errs, lens = np.array(errs, float), np.array(lens, float)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    w = errs[idx].sum(1) / lens[idx].sum(1) * 100
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))


def paired_bootstrap(cache, a_scores, b_scores, alpha=ALPHA, B=5000, seed=0,
                     alpha_a=None, alpha_b=None, subset=None):
    """Is arm A's WER really below arm B's? Resample sentences, keep both arms paired.

    Each arm is evaluated at its own development-tuned interpolation weight, so the test
    compares the arms as the results table reports them rather than at a shared weight
    that may suit one score scale and not the other.
    """
    pa = picks(cache, a_scores, alpha_a if alpha_a is not None else alpha)
    pb = picks(cache, b_scores, alpha_b if alpha_b is not None else alpha)
    # Significance must be measured on the same held-out split the table reports, or it
    # silently includes the development sentences the weights were tuned on.
    keys = [s for s in (subset if subset is not None else pa) if s in pa and s in pb]
    ea, eb, ln = [], [], []
    for s in keys:
        ma = jiwer.process_words([s], [pa[s]])
        mb = jiwer.process_words([s], [pb[s]])
        ea.append(ma.substitutions + ma.deletions + ma.insertions)
        eb.append(mb.substitutions + mb.deletions + mb.insertions)
        ln.append(len(s.split()))
    ea, eb, ln = np.array(ea, float), np.array(eb, float), np.array(ln, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    d = (ea[idx].sum(1) / ln[idx].sum(1)) - (eb[idx].sum(1) / ln[idx].sum(1))
    obs = ea.sum() / ln.sum() - eb.sum() / ln.sum()
    # one-sided: how often does the resampled difference fail to favour A
    return {"delta_wer": float(obs) * 100, "p": float(np.mean(d >= 0)),
            "ci": (float(np.percentile(d, 2.5)) * 100, float(np.percentile(d, 97.5)) * 100)}


def alpha_sweep(cache, scores, alphas=None):
    alphas = alphas if alphas is not None else np.round(np.arange(0.0, 1.01, 0.05), 2)
    return [(float(a), metrics(cache, scores, alpha=a)["wer"]) for a in alphas]


# ---- calibration -------------------------------------------------------------------

def _correctness(cache, scores, conf, alpha=ALPHA):
    p = picks(cache, scores, alpha)
    keys = [s for s in p if conf.get(s) is not None]
    c = np.array([conf[s] for s in keys], float)
    ok = np.array([p[s] == s for s in keys], bool)
    return c, ok


def ece(conf, correct, bins=10):
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    n, tot, mx = len(conf), 0.0, 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            rows.append((float(lo), float(hi), 0, None, None))
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        gap = abs(acc - cf)
        tot += m.sum() / n * gap
        mx = max(mx, gap)
        rows.append((float(lo), float(hi), int(m.sum()), float(cf), float(acc)))
    return {"ece": float(tot), "mce": float(mx), "bins": rows}


def coverage_risk(conf, correct):
    """Sort by confidence and walk down: what error rate at what coverage if we abstain."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    risk = 1 - np.cumsum(ok) / np.arange(1, len(c) + 1)
    return cov, risk, c


def load(path):
    return pickle.load(open(path, "rb"))


def load_lists(path):
    return pickle.load(open(path, "rb"))
