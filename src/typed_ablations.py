"""Two robustness checks on the typed-selection negative result.

Q1 (permutation): if the failure is label-position collapse, then shuffling the menu and
aggregating across shuffles should recover some of the lost accuracy. We use Borda count
over `k` random permutations - the cheap version of permutation self-consistency.

Q6 (menu size): if collapse is driven by menu length, a shorter menu should suffer less.
We sweep N over the top-N candidates by first-pass score.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

import analyze as A


def borda(rank_lists, n):
    """Aggregate several orderings into one score vector."""
    pts = np.zeros(n)
    for order in rank_lists:
        for rank, idx in enumerate(order):
            pts[idx] += n - rank
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["optchoice", "qwenchoice"])
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from runscores_cuda import build
    cache, meta = pickle.load(open(os.path.expanduser("~/b2t/work/lists_real.pkl"), "rb"))
    sents = list(cache)[: a.limit]
    rng = np.random.default_rng(0)
    res = {"arm": a.arm, "n_sentences": len(sents)}

    # ---- Q6: menu size ----------------------------------------------------------
    res["menu_size"] = {}
    for N in (5, 10, 20, 60):
        scorer = build(a.arm, N)
        hit = 0
        used = set()
        for s in sents:
            hyps = [h for h, _, _ in cache[s]][:N]
            if len(hyps) < 2:
                continue
            sc = scorer(s, hyps)[: len(hyps)]
            k = int(np.argmax(sc))
            used.add(k)
            if hyps[k] == s:
                hit += 1
        res["menu_size"][N] = {"truth_picked_pct": hit / len(sents) * 100,
                               "distinct_labels": len(used)}
        print(f"  N={N:3}  truth picked {hit/len(sents)*100:5.1f}%  "
              f"labels used {len(used)}", flush=True)

    # ---- Q1: shuffle + Borda ----------------------------------------------------
    scorer = build(a.arm, meta["nbest"])
    base_hit, borda_hit, t0 = 0, 0, time.perf_counter()
    for s in sents:
        hyps = [h for h, _, _ in cache[s]]
        if len(hyps) < 2:
            continue
        sc = scorer(s, hyps)[: len(hyps)]
        if hyps[int(np.argmax(sc))] == s:
            base_hit += 1
        orders = []
        for _ in range(a.shuffles):
            perm = rng.permutation(len(hyps))
            shuffled = [hyps[i] for i in perm]
            ssc = np.asarray(scorer(s, shuffled)[: len(hyps)])
            order_in_shuffled = np.argsort(ssc)[::-1]
            orders.append([int(perm[i]) for i in order_in_shuffled])
        pts = borda(orders, len(hyps))
        if hyps[int(np.argmax(pts))] == s:
            borda_hit += 1
    res["permutation"] = {
        "shuffles": a.shuffles,
        "baseline_truth_picked_pct": base_hit / len(sents) * 100,
        "borda_truth_picked_pct": borda_hit / len(sents) * 100,
        "wall_s": time.perf_counter() - t0,
    }
    print(f"  single order: {base_hit/len(sents)*100:.1f}%   "
          f"Borda over {a.shuffles} shuffles: {borda_hit/len(sents)*100:.1f}%", flush=True)

    out = a.out or os.path.expanduser(f"~/b2t/work/ablation_{a.arm}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
