"""Label-position diagnostic for typed arms: which label each arm picks, before any combination.

    python collapse_diag.py --lists LISTS.pkl --scores DIR --prefix kaldi_
Reproduces the T15 short paper's Table 3 (labels used, share on A, truth picked, truth picked when not A).
"""
import argparse, os, pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--lists", required=True); ap.add_argument("--scores", required=True)
ap.add_argument("--prefix", default="t12_"); ap.add_argument("--arms", default="jev,laya,qwenchoice,optchoice")
a = ap.parse_args()
cache, meta = pickle.load(open(a.lists, "rb"))
top1 = np.mean([nb[0][0] == ref for ref, nb in cache.items()])
print(f"n lists {len(cache)} | decoder top-1 sentence accuracy {top1*100:.1f}%")
for arm in a.arms.split(","):
    f = os.path.join(a.scores, f"{a.prefix}{arm}.pkl")
    if not os.path.exists(f):
        print(f"{arm:11s} (no scores)"); continue
    sc = pickle.load(open(f, "rb"))["scores"]
    picks, hit, dev_hit, dev_n = [], 0, 0, 0
    for ref, nb in cache.items():
        if ref not in sc: continue
        j = int(np.argmax(sc[ref])); picks.append(j)
        ok = nb[j][0] == ref; hit += ok
        if j != 0: dev_n += 1; dev_hit += ok
    picks = np.array(picks)
    print(f"{arm:11s} labels used {len(set(picks.tolist()))}/{meta['nbest']} | share on A {np.mean(picks==0)*100:.0f}% | "
          f"truth picked {hit/len(picks)*100:.1f}% | truth picked when not A {dev_hit/max(dev_n,1)*100:.1f}% (n={dev_n})")
