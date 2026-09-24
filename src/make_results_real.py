"""Everything the real-data paper reports, computed from the cached arm scores.

The interpolation weight is tuned per arm on a development split and applied to a disjoint
test split, so tuning it per rescorer does not become another way of leaking the answer.
"""
import glob
import json
import os
import pickle

import jiwer
import numpy as np

import analyze as A

DEV_FRAC = 0.30
SCORES = os.path.expanduser("~/b2t/work/scores")


def split(cache, seed=0):
    keys = sorted(cache)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(keys))
    k = int(len(keys) * DEV_FRAC)
    return [keys[i] for i in idx[:k]], [keys[i] for i in idx[k:]]


def tune_alpha(cache, scores, dev):
    best, best_w = 0.55, 1e9
    for a in np.round(np.arange(0.0, 1.001, 0.05), 2):
        w = A.metrics(cache, scores, alpha=float(a), subset=dev)["wer"]
        if w < best_w:
            best, best_w = float(a), w
    return best


def arm_row(cache, rec, dev, test):
    sc = rec["scores"] if rec else None
    alpha = tune_alpha(cache, sc, dev) if sc else 0.55
    tuned = A.metrics(cache, sc, alpha=alpha, subset=test)
    fixed = A.metrics(cache, sc, alpha=0.55, subset=test)
    lo, hi = A.bootstrap_wer({k: cache[k] for k in test}, sc, alpha=alpha)
    lat = list(rec["latency"].values()) if rec else []
    ptok = [v for v in (rec["prompt_tokens"].values() if rec else []) if v]
    return {"label": rec["label"] if rec else "No rescoring (first pass only)",
            "arm": rec["arm"] if rec else "none",
            "wer": tuned["wer"], "wer_fixed": fixed["wer"], "wer_ci": [lo, hi],
            "sent_acc": tuned["sent_acc"], "alpha": alpha, "n": tuned["n"],
            "median_ms": float(np.median(lat)) if lat else None,
            "p90_ms": float(np.percentile(lat, 90)) if lat else None,
            "median_prompt_tokens": float(np.median(ptok)) if ptok else None}


def calibration(cache, rec, alpha):
    if not rec or not any(v is not None for v in rec["conf"].values()):
        return None
    c, ok = A._correctness(cache, rec["scores"], rec["conf"], alpha)
    if len(c) == 0:
        return None
    e = A.ece(c, ok)
    cov, risk, thr = A.coverage_risk(c, ok)
    pts = []
    for t in (0.3, 0.5, 0.7, 0.9, 1.0):
        i = min(int(t * len(cov)) - 1, len(cov) - 1)
        if i >= 0:
            pts.append({"coverage": float(cov[i]), "risk": float(risk[i]),
                        "threshold": float(thr[i])})
    return {"ece": e["ece"], "mce": e["mce"], "bins": e["bins"],
            "mean_conf_right": float(c[ok].mean()) if ok.any() else None,
            "mean_conf_wrong": float(c[~ok].mean()) if (~ok).any() else None,
            "n_right": int(ok.sum()), "n_wrong": int((~ok).sum()),
            "coverage_risk": pts}


def main():
    cache, meta = pickle.load(open(os.path.expanduser("~/b2t/work/lists_real.pkl"), "rb"))
    dev, test = split(cache)
    R = {"meta": meta, "n_dev": len(dev), "n_test": len(test), "arms": [], "calibration": {}}

    R["arms"].append(arm_row(cache, None, dev, test))
    have = {}
    for f in sorted(glob.glob(os.path.join(SCORES, "real_*.pkl"))):
        rec = pickle.load(open(f, "rb"))
        if len(rec["scores"]) < len(cache) * 0.9:
            print(f"  (skipping partial arm {rec['arm']}: {len(rec['scores'])} sentences)")
            continue
        row = arm_row(cache, rec, dev, test)
        R["arms"].append(row)
        have[rec["arm"]] = rec["scores"]
        cb = calibration(cache, rec, row["alpha"])
        if cb:
            R["calibration"][rec["arm"]] = cb
    R["arms"].sort(key=lambda r: r["wer"])

    orc = [min(jiwer.wer(s, h) for h, _, _ in cache[s]) for s in test]
    R["oracle_wer"] = float(np.mean(orc)) * 100
    R["truth_in_nbest"] = float(np.mean(
        [s in [h for h, _, _ in cache[s]] for s in test])) * 100

    print(f"\n=== REAL T15 DATA  (n_test={R['n_test']}, oracle {R['oracle_wer']:.1f}%, "
          f"truth-in-list {R['truth_in_nbest']:.0f}%) ===")
    for a in R["arms"]:
        ms = f"{a['median_ms']:.0f}" if a["median_ms"] else "-"
        print(f"  {a['label'][:46]:<48} WER {a['wer']:5.1f} "
              f"[{a['wer_ci'][0]:4.1f},{a['wer_ci'][1]:4.1f}]  "
              f"acc {a['sent_acc']:5.1f}  a={a['alpha']:.2f}  {ms:>6}ms")

    # Every typed arm against every per-candidate arm, paired on the same sentences.
    sig = {}
    typed = [k for k in have if k.endswith("choice") or k == "jev"]
    percand = [k for k in have if not (k.endswith("choice") or k == "jev")]
    for a in typed:
        for b in percand:
            sig[f"{a}_vs_{b}"] = A.paired_bootstrap(cache, have[a], have[b])
    # And each model against itself in the other shape.
    for m in ("opt", "gpt2", "qwen"):
        if m in have and f"{m}choice" in have:
            sig[f"{m}choice_vs_{m}"] = A.paired_bootstrap(cache, have[f"{m}choice"], have[m])
    R["significance"] = sig
    if sig:
        print("\n=== paired bootstrap (A - B, negative favours A) ===")
        for k, v in sorted(sig.items()):
            star = "*" if v["p"] < 0.05 else " "
            print(f" {star}{k:<26} {v['delta_wer']:+6.2f}  "
                  f"[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]  p={v['p']:.4f}")

    if R["calibration"]:
        print("\n=== calibration ===")
        for arm, c in R["calibration"].items():
            print(f"  {arm:<12} ECE {c['ece']:.3f}  MCE {c['mce']:.3f}  "
                  f"conf right {c['mean_conf_right']:.3f} / wrong {c['mean_conf_wrong']:.3f} "
                  f"(n={c['n_right']}/{c['n_wrong']})")

    json.dump(R, open(os.path.expanduser("~/b2t/work/results_real.json"), "w"), indent=2)
    print("\nwrote results_real.json")


if __name__ == "__main__":
    main()
