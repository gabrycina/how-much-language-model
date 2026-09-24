"""Analyses requested at review, all computed from cached scores - no model re-runs.

The substantive one is the softmax-over-candidates baseline: if a per-candidate scorer's
scores can be turned into a calibrated selection confidence, then abstention is not a
property of typed decisions at all, and our own observation is weaker than we claimed.
"""
import glob
import json
import pickle

import numpy as np

import analyze as A

DEV_FRAC = 0.30


def split(cache, seed=0):
    keys = sorted(cache)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(keys))
    k = int(len(keys) * DEV_FRAC)
    return [keys[i] for i in idx[:k]], [keys[i] for i in idx[k:]]


def softmax_conf(scores, T):
    v = np.asarray(scores, dtype=float) / T
    e = np.exp(v - v.max())
    p = e / e.sum()
    return float(p.max()), int(np.argmax(v))


def fit_temperature(cache, rec, dev, alpha):
    """Pick T on dev by expected calibration error, as the reviewer suggests."""
    best, best_ece = 1.0, 9e9
    picks = A.picks(cache, rec["scores"], alpha)
    for T in [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]:
        cs, oks = [], []
        for s in dev:
            if s not in rec["scores"]:
                continue
            c, _ = softmax_conf(rec["scores"][s], T)
            cs.append(c)
            oks.append(picks[s] == s)
        if not cs:
            continue
        e = A.ece(np.array(cs), np.array(oks))["ece"]
        if e < best_ece:
            best, best_ece = T, e
    return best, best_ece


def curve(cache, rec, keys, alpha, T):
    picks = A.picks(cache, rec["scores"], alpha)
    cs, oks = [], []
    for s in keys:
        if s not in rec["scores"]:
            continue
        c, _ = softmax_conf(rec["scores"][s], T)
        cs.append(c)
        oks.append(picks[s] == s)
    c, ok = np.array(cs), np.array(oks)
    cov, risk, thr = A.coverage_risk(c, ok)
    pts = {}
    for t in (0.3, 0.5, 0.7, 0.9, 1.0):
        i = min(int(t * len(cov)) - 1, len(cov) - 1)
        pts[int(round(t * 100))] = float(risk[i]) * 100
    return {"ece": A.ece(c, ok)["ece"], "temperature": T, "curve": pts,
            "conf_right": float(c[ok].mean()) if ok.any() else None,
            "conf_wrong": float(c[~ok].mean()) if (~ok).any() else None}


def main():
    cache, meta = pickle.load(open("real/lists_real.pkl", "rb"))
    dev, test = split(cache)
    R = json.load(open("real/results_final.json"))
    alphas = {r["arm"]: r["alpha"] for r in R["rows"]}
    out = {}

    # --- Q3: token accounting, measured rather than assumed -------------------------
    import string
    per_cand_chars = [sum(len(h) for h, _, _ in v) for v in cache.values()]
    n_cand = [len(v) for v in cache.values()]
    out["tokens"] = {
        "mean_candidates_per_list": float(np.mean(n_cand)),
        "mean_chars_per_candidate": float(np.mean(
            [c / n for c, n in zip(per_cand_chars, n_cand)])),
        "per_candidate_tokens_per_decision_est": float(np.mean(per_cand_chars) / 4),
        "note": "~4 chars/token; per-candidate reads every candidate once, "
                "typed reads the same text plus instructions and labels",
    }
    for f in sorted(glob.glob("real/scores/real_*.pkl")):
        rec = pickle.load(open(f, "rb"))
        pt = [v for v in rec["prompt_tokens"].values() if v]
        if pt:
            out["tokens"][f"{rec['arm']}_prompt_tokens_median"] = float(np.median(pt))

    # --- Q4: is the OPT / GPT-2 gap significant? ------------------------------------
    have = {}
    for f in sorted(glob.glob("real/scores/real_*.pkl")):
        rec = pickle.load(open(f, "rb"))
        if len(rec["scores"]) >= len(cache) * 0.9:
            have[rec["arm"]] = rec["scores"]
    pairs = [("gpt2", "opt"), ("qwen", "opt"), ("jev", "opt"), ("jev", "gpt2"),
             ("laya", "gpt2"), ("optchoice", "opt"), ("qwenchoice", "qwen")]
    out["significance"] = {}
    for a, b in pairs:
        if a in have and b in have:
            out["significance"][f"{a}_vs_{b}"] = A.paired_bootstrap(
                cache, have[a], have[b],
                alpha_a=alphas.get(a, 0.55), alpha_b=alphas.get(b, 0.55), subset=test)

    # --- Q8: can per-candidate scorers abstain too? ---------------------------------
    out["softmax_abstention"] = {}
    for f in sorted(glob.glob("real/scores/real_*.pkl")):
        rec = pickle.load(open(f, "rb"))
        if len(rec["scores"]) < len(cache) * 0.9:
            continue
        alpha = alphas.get(rec["arm"], 0.55)
        T, dev_ece = fit_temperature(cache, rec, dev, alpha)
        out["softmax_abstention"][rec["arm"]] = curve(cache, rec, test, alpha, T)

    json.dump(out, open("real/revision_analysis.json", "w"), indent=2)

    print("=== Q3  token accounting ===")
    for k, v in out["tokens"].items():
        print(f"  {k}: {v}")
    print("\n=== Q4  paired bootstrap, explicit p-values ===")
    for k, v in out["significance"].items():
        print(f"  {k:<22} dWER {v['delta_wer']:+6.2f}  "
              f"[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]  p={v['p']:.4f}")
    print("\n=== Q8  softmax-over-candidates abstention (temperature fit on dev) ===")
    print(f"  {'arm':<12}{'T':>5}{'ECE':>7}   coverage->error")
    for arm, c in out["softmax_abstention"].items():
        pts = "  ".join(f"{k}%:{v:.0f}%" for k, v in c["curve"].items())
        print(f"  {arm:<12}{c['temperature']:>5}{c['ece']:>7.3f}   {pts}")
    print("\nwrote real/revision_analysis.json")


if __name__ == "__main__":
    main()
