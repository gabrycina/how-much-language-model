"""Every number in the short paper (paper-short/): Figure 1, Tables 1, 2 and 4.

Runs from cached lists, per-arm scores and the latency logs; no GPU or API needed.

    python src/short_paper_results.py
"""
import json, os, pickle
import numpy as np, jiwer
import analyze as A
from revision_analysis import split

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
cache, _ = pickle.load(open(os.path.join(ROOT, "data", "lists_published.pkl"), "rb"))
dev, test = split(cache)
L = lambda a: pickle.load(open(os.path.join(ROOT, "data", "scores_published", f"kaldi_{a}.pkl"), "rb"))
_e = {}
def err(r, h):
    if (r, h) not in _e:
        m = jiwer.process_words([r], [h]); _e[(r, h)] = m.substitutions + m.deletions + m.insertions
    return _e[(r, h)]
ln = np.array([len(s.split()) for s in test], float)
ix2k = np.random.default_rng(0).integers(0, len(test), (2000, len(test)))
ix5k = np.random.default_rng(0).integers(0, len(test), (5000, len(test)))
zero = {s: np.zeros(len(nb)) for s, nb in cache.items()}

# (alpha, lambda) per arm and protocol, as selected on the dev split (src/joint_tuning.py)
CFG = {"No rescoring": ("none", {"published": (0, .3), "retuned": (0, .3)}),
       "OPT-6.7b": ("opt", {"published": (.30, .3), "retuned": (.45, .40)}),
       "Qwen2.5-7B": ("qwen", {"published": (.45, .3), "retuned": (.45, .45)}),
       "Jev (typed)": ("jev", {"published": (.90, .3), "retuned": (.75, 2.0)})}
errs, out = {}, {}
for name, (arm, prots) in CFG.items():
    sc = zero if arm == "none" else L(arm)["scores"]
    for prot, (a, lam) in prots.items():
        p = A.picks(cache, sc, a, acoustic_scale=lam)
        ev = np.array([err(s, p[s]) for s in test], float); errs[(name, prot)] = ev
        bs = ev[ix2k].sum(1) / ln[ix2k].sum(1) * 100
        out.setdefault(name, {})[prot] = {"wer": ev.sum() / ln.sum() * 100, "lo": float(np.percentile(bs, 2.5)),
                                          "hi": float(np.percentile(bs, 97.5)), "alpha": a, "lambda": lam}
        print(f"{name:13s} {prot:9s} WER {out[name][prot]['wer']:.2f} [{out[name][prot]['lo']:.2f}, {out[name][prot]['hi']:.2f}]")

def paired(a, b):
    d = (a[ix5k].sum(1) - b[ix5k].sum(1)) / ln[ix5k].sum(1) * 100
    return (a.sum() - b.sum()) / ln.sum() * 100, np.percentile(d, [2.5, 97.5]), float(np.mean(d >= 0))
print("\nTable 2: Jev minus each 7B model")
for prot in ("published", "retuned"):
    for other in ("OPT-6.7b", "Qwen2.5-7B"):
        dlt, ci, p = paired(errs[("Jev (typed)", prot)], errs[(other, prot)])
        print(f"  {prot:9s} Jev - {other:10s} {dlt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.3f}")

print("\nTable 4: cost and latency")
runs = [json.load(open(os.path.join(ROOT, "analysis", f"jev_latency_run{i}.json"))) for i in (1, 2)]
tok = np.array(runs[0]["tokens"] + [r["tok"] for r in runs[1]["rows"]], float)
jev_1k = tok.mean() * 0.042e-6 * 1000
print(f"  Jev: {len(tok)} lists, mean {tok.mean():.0f} input tokens -> ${jev_1k:.3f} per 1k sentences")
for arm, price in (("opt", 1.82), ("qwen", 1.82), ("jev", None)):
    lat = np.array([v for v in L(arm)["latency"].values() if v is not None], float)
    line = f"  {arm:5s} latency median {np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms"
    if price:
        gpu_1k = price / 3600 * lat.mean() / 1000 * 1000
        line += f"; ${gpu_1k:.3f} per 1k at full use; break-even utilisation {gpu_1k / jev_1k:.0%}"
    print(line)
json.dump(out, open(os.path.join(ROOT, "analysis", "short_paper_fig1.json"), "w"), indent=1)
