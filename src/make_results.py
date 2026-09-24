"""Everything the paper reports, computed from the cached scores.

The interpolation weight is tuned on a development split and applied to a disjoint test
split, so that tuning it per arm - which a reviewer asked for - does not quietly become a
fourth way of leaking the answer.
"""
import glob
import json
import os
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


def tune_alpha(cache, scores, dev, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.0, 1.001, 0.05), 2)
    best, best_w = 0.55, 1e9
    for a in grid:
        w = A.metrics(cache, scores, alpha=float(a), subset=dev)["wer"]
        if w < best_w:
            best, best_w = float(a), w
    return best


def arm_row(cache, rec, dev, test):
    sc = rec["scores"] if rec else None
    a_tuned = tune_alpha(cache, sc, dev) if sc else 0.55
    fixed = A.metrics(cache, sc, alpha=0.55, subset=test)
    tuned = A.metrics(cache, sc, alpha=a_tuned, subset=test)
    lo, hi = A.bootstrap_wer({k: cache[k] for k in test}, sc, alpha=a_tuned)
    lat = list(rec["latency"].values()) if rec else []
    ptok = [v for v in (rec["prompt_tokens"].values() if rec else []) if v]
    return {
        "label": rec["label"] if rec else "No rescoring (n-gram only)",
        "arm": rec["arm"] if rec else "none",
        "wer_fixed": fixed["wer"], "wer": tuned["wer"], "wer_ci": [lo, hi],
        "sent_acc": tuned["sent_acc"], "alpha": a_tuned, "n": tuned["n"],
        "median_ms": float(np.median(lat)) if lat else None,
        "p90_ms": float(np.percentile(lat, 90)) if lat else None,
        "median_prompt_tokens": float(np.median(ptok)) if ptok else None,
    }


def calibration_block(cache, rec, alpha):
    if not rec or not any(v is not None for v in rec["conf"].values()):
        return None
    c, ok = A._correctness(cache, rec["scores"], rec["conf"], alpha)
    if len(c) == 0:
        return None
    e = A.ece(c, ok)
    cov, risk, thr = A.coverage_risk(c, ok)
    pts = []
    for target in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        i = min(int(target * len(cov)) - 1, len(cov) - 1)
        if i >= 0:
            pts.append({"coverage": float(cov[i]), "risk": float(risk[i]),
                        "threshold": float(thr[i])})
    return {"ece": e["ece"], "mce": e["mce"], "bins": e["bins"],
            "mean_conf_right": float(c[ok].mean()) if ok.any() else None,
            "mean_conf_wrong": float(c[~ok].mean()) if (~ok).any() else None,
            "n_right": int(ok.sum()), "n_wrong": int((~ok).sum()),
            "coverage_risk": pts}


def run_regime(lists_path, tag):
    cache, meta = pickle.load(open(lists_path, "rb"))
    dev, test = split(cache)
    out = {"meta": meta, "n_dev": len(dev), "n_test": len(test), "arms": [], "calibration": {}}

    out["arms"].append(arm_row(cache, None, dev, test))
    for f in sorted(glob.glob(f"scores/{tag}_*.pkl")):
        rec = pickle.load(open(f, "rb"))
        # Subset arms (the denoising ablation) are reported separately.
        if len(rec["scores"]) < len(cache) * 0.9:
            continue
        row = arm_row(cache, rec, dev, test)
        out["arms"].append(row)
        cb = calibration_block(cache, rec, row["alpha"])
        if cb:
            out["calibration"][rec["arm"]] = cb
    out["arms"].sort(key=lambda r: r["wer"])

    # Oracle floor on the test split.
    import jiwer
    orc = [min(jiwer.wer(s, h) for h, _, _ in cache[s]) for s in test]
    out["oracle_wer"] = float(np.mean(orc)) * 100
    out["truth_in_nbest"] = float(np.mean([s in [h for h, _, _ in cache[s]] for s in test])) * 100
    return out, cache, dev, test


def main():
    results = {}
    for tag, path in [("easy", "lists_easy_s0.pkl"), ("hard", "lists_hard_s0.pkl")]:
        p = f"lists_{tag}_s0.pkl"
        if not os.path.exists(p):
            continue
        r, cache, dev, test = run_regime(p, f"{tag}_s0")
        results[tag] = r
        print(f"\n=== {tag}  (oracle {r['oracle_wer']:.1f}%, "
              f"truth-in-list {r['truth_in_nbest']:.0f}%, n_test={r['n_test']}) ===")
        for a in r["arms"]:
            ci = f"[{a['wer_ci'][0]:.1f},{a['wer_ci'][1]:.1f}]"
            ms = f"{a['median_ms']:.0f}" if a["median_ms"] else "-"
            print(f"  {a['label'][:46]:<48} WER {a['wer']:5.1f} {ci:<14} "
                  f"acc {a['sent_acc']:5.1f}  a={a['alpha']:.2f}  {ms:>6}ms")

    # Denoising-step ablation (subset arms).
    abl = []
    if os.path.exists("lists_easy_s0.pkl"):
        cache, _ = pickle.load(open("lists_easy_s0.pkl", "rb"))
        for f in sorted(glob.glob("scores/easy_s0_dg*.pkl")):
            rec = pickle.load(open(f, "rb"))
            sub = list(rec["scores"])
            if len(sub) >= len(cache) * 0.9 and rec["arm"] not in ("dgscore1",):
                continue
            m = A.metrics(cache, rec["scores"], alpha=0.55, subset=sub)
            abl.append({"arm": rec["arm"], "label": rec["label"], "n": len(sub),
                        "wer": m["wer"], "sent_acc": m["sent_acc"],
                        "median_ms": float(np.median(list(rec["latency"].values())))})
        abl.sort(key=lambda r: r["arm"])
        if abl:
            print("\n=== per-candidate diffusion ablation (subset) ===")
            for a in abl:
                print(f"  {a['label'][:52]:<54} n={a['n']:3}  WER {a['wer']:5.1f}  "
                      f"{a['median_ms']:6.0f}ms")
    results["ablation"] = abl

    # Alpha sensitivity on the easy regime.
    sens = {}
    if os.path.exists("lists_easy_s0.pkl"):
        cache, _ = pickle.load(open("lists_easy_s0.pkl", "rb"))
        for f in sorted(glob.glob("scores/easy_s0_*.pkl")):
            rec = pickle.load(open(f, "rb"))
            if len(rec["scores"]) < len(cache) * 0.9:
                continue
            sens[rec["arm"]] = A.alpha_sweep(cache, rec["scores"])
    results["alpha_sensitivity"] = sens

    # Seed variance on the headline arms.
    seeds = {}
    for s in (0, 1, 2):
        p = f"lists_easy_s{s}.pkl"
        if not os.path.exists(p):
            continue
        cache, _ = pickle.load(open(p, "rb"))
        for arm in ("jev", "dg"):
            f = f"scores/easy_s{s}_{arm}.pkl"
            if not os.path.exists(f):
                continue
            rec = pickle.load(open(f, "rb"))
            m = A.metrics(cache, rec["scores"], alpha=0.55)
            seeds.setdefault(arm, []).append({"seed": s, "wer": m["wer"],
                                              "sent_acc": m["sent_acc"]})
    results["seeds"] = seeds
    if seeds:
        print("\n=== seed variance (fixed alpha) ===")
        for arm, rows in seeds.items():
            w = [r["wer"] for r in rows]
            print(f"  {arm:<12} WER {np.mean(w):.1f} +/- {np.std(w):.1f} over {len(w)} seeds")

    # Paired significance against the strongest non-typed arm.
    sig = {}
    if os.path.exists("lists_easy_s0.pkl"):
        cache, _ = pickle.load(open("lists_easy_s0.pkl", "rb"))
        have = {}
        for f in glob.glob("scores/easy_s0_*.pkl"):
            rec = pickle.load(open(f, "rb"))
            if len(rec["scores"]) >= len(cache) * 0.9:
                have[rec["arm"]] = rec["scores"]
        for a, b in [("jev", "ar7b"), ("dg", "ar7b"), ("jev", "gpt2"), ("dg", "dgscore1"),
                     ("ar7bchoice", "ar7b")]:
            if a in have and b in have:
                sig[f"{a}_vs_{b}"] = A.paired_bootstrap(cache, have[a], have[b])
    results["significance"] = sig
    if sig:
        print("\n=== paired bootstrap (WER difference, A - B) ===")
        for k, v in sig.items():
            print(f"  {k:<22} {v['delta_wer']:+6.2f}  95% CI "
                  f"[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]  p={v['p']:.4f}")

    json.dump(results, open("results_full.json", "w"), indent=2)
    print("\nwrote results_full.json")


if __name__ == "__main__":
    main()
