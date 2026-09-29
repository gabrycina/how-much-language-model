"""The pre-registered analysis (PREREGISTRATION_T12.md), runnable on any participant's lists.

    python prereg_analysis.py --lists LISTS.pkl --scores DIR --prefix t12_ --train-text TRAIN.txt --out OUT.json

On T15 (`--lists ../data/lists_published.pkl --scores ../data/scores_published --prefix kaldi_`) it
reproduces the short paper's Tables 1-2, which is how this code was checked before T12 was scored.
"""
import argparse, json, os, pickle, re
import numpy as np, jiwer
import analyze as A
from revision_analysis import split

LAMBDAS = [0, .1, .2, .25, .3, .35, .4, .45, .5, .6, .8, 1.0, 1.25, 1.5, 2, 3, 5]
ALPHAS = [float(a) for a in np.round(np.arange(0, 1.001, .05), 2)]
MAIN = ["opt", "qwen", "jev"]
CONTROLS = ["optchoice", "qwenchoice", "laya"]
_P = str.maketrans("", "", ",.?!-’'\";:()[]")
norm = lambda t: " ".join(t.translate(_P).lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", required=True); ap.add_argument("--scores", required=True)
    ap.add_argument("--prefix", default="t12_"); ap.add_argument("--train-text", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache, meta = pickle.load(open(a.lists, "rb")); dev, test = split(cache)
    E = {}
    def err(r, h):
        if (r, h) not in E:
            m = jiwer.process_words([r], [h]); E[(r, h)] = m.substitutions + m.deletions + m.insertions
        return E[(r, h)]
    def wer(p, ks): return 100 * sum(err(s, p[s]) for s in ks) / sum(len(s.split()) for s in ks)
    zero = {s: np.zeros(len(nb)) for s, nb in cache.items()}
    def load(arm):
        f = os.path.join(a.scores, f"{a.prefix}{arm}.pkl")
        if not os.path.exists(f): return None
        raw = pickle.load(open(f, "rb"))["scores"]
        return {s: raw.get(s, zero[s]) for s in cache}   # unscored (single-candidate) lists: first pass
    picks = lambda sc, al, lam: A.picks(cache, sc, al, acoustic_scale=lam)

    lam_A = min(LAMBDAS, key=lambda l: wer(picks(zero, 0.0, l), dev))
    fp_dev = wer(picks(zero, 0.0, lam_A), dev); delta = 0.025 * fp_dev
    fp = picks(zero, 0.0, lam_A)
    out = {"n_dev": len(dev), "n_test": len(test), "list_meta": {k: v for k, v in meta.items() if k != "source"},
           "lambda_A": lam_A, "first_pass_dev_wer": fp_dev, "first_pass_test_wer": wer(fp, test),
           "margin_delta": delta, "arms": {}}

    def evaluate(subset, label):
        ln = np.array([len(s.split()) for s in subset], float)
        ix = np.random.default_rng(0).integers(0, len(subset), (5000, len(subset)))
        ev = lambda p: np.array([err(s, p[s]) for s in subset], float)
        base = ev(fp); res = {"n": len(subset), "arms": {}, "primary": {}}
        chosen = {}
        for arm in MAIN + CONTROLS:
            sc = load(arm)
            if sc is None: continue
            for prot in ("A", "B"):
                if prot == "A":
                    lam = lam_A; al = min(ALPHAS, key=lambda x: wer(picks(sc, x, lam_A), dev))
                else:
                    _, lam, al = min((wer(picks(sc, x, l), dev), l, x) for l in LAMBDAS for x in ALPHAS)
                p = picks(sc, al, lam); e = ev(p); chosen[(arm, prot)] = e
                d = (e[ix].sum(1) - base[ix].sum(1)) / ln[ix].sum(1) * 100
                res["arms"][f"{arm}/{prot}"] = {"lambda": lam, "alpha": al, "wer": e.sum() / ln.sum() * 100,
                                               "delta_vs_none": (e.sum() - base.sum()) / ln.sum() * 100,
                                               "p_vs_none": float(np.mean(d >= 0))}
        for prot in ("A", "B"):   # Holm over the three main arms vs none, within protocol
            ps = sorted((res["arms"][f"{m}/{prot}"]["p_vs_none"], m) for m in MAIN if f"{m}/{prot}" in res["arms"])
            run = 0.0
            for i, (p, m) in enumerate(ps):
                run = max(run, min(1.0, (len(ps) - i) * p)); res["arms"][f"{m}/{prot}"]["p_holm"] = run
            for m in ("opt", "qwen"):
                if ("jev", prot) in chosen and (m, prot) in chosen:
                    d = (chosen[("jev", prot)][ix].sum(1) - chosen[(m, prot)][ix].sum(1)) / ln[ix].sum(1) * 100
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    diff = (chosen[("jev", prot)].sum() - chosen[(m, prot)].sum()) / ln.sum() * 100
                    res["primary"][f"jev-{m}/{prot}"] = {"diff": diff, "ci": [lo, hi],
                                                        "matches_relative": bool(hi < delta),
                                                        "matches_abs_0.2": bool(hi < 0.2)}
        verdicts = [v["matches_relative"] for v in res["primary"].values()]
        res["verdict"] = ("confirmed" if verdicts and all(verdicts) else
                          "partly confirmed" if any(verdicts) else "not confirmed")
        print(f"[{label}] n={len(subset)} verdict: {res['verdict']}")
        for k, v in res["primary"].items():
            print(f"   {k:14s} diff {v['diff']:+.2f}  95% CI [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]  "
                  f"< delta {delta:.3f}: {v['matches_relative']}   < 0.2: {v['matches_abs_0.2']}")
        return res

    print(f"lambda_A {lam_A} | first pass dev {fp_dev:.2f}% test {out['first_pass_test_wer']:.2f}% | delta {delta:.3f}")
    out["all"] = evaluate(test, "all test")
    for k, v in out["all"]["arms"].items():
        print(f"   {k:14s} lam {v['lambda']:<5} a {v['alpha']:<5} WER {v['wer']:.2f}  vs none {v['delta_vs_none']:+.2f}"
              f"  p {v['p_vs_none']:.4f}" + (f"  holm {v['p_holm']:.4f}" if "p_holm" in v else ""))
    if a.train_text:
        tr = set(norm(l) for l in open(a.train_text))
        unseen = [s for s in test if s not in tr]
        out["n_test_seen_in_train"] = len(test) - len(unseen)
        out["unseen"] = evaluate(unseen, f"test without {len(test)-len(unseen)} train-overlap sentences")
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
