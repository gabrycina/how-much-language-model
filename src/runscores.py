"""Run one rescoring arm over a cached set of n-best lists and save its raw scores.

Separating scoring from analysis is what makes the rest of the evaluation affordable: the
interpolation sweep, the bootstrap intervals and every calibration metric are computed
from these files without touching a model again.
"""
import argparse
import os
import pickle
import time

import numpy as np

ARMS = {
    "jev":       ("Jev (typed, hosted API)", lambda n: _jev(n)),
    "dg":        ("DiffusionGemma 26B (typed, local)", lambda n: _dg(n)),
    "dgscore1":  ("DiffusionGemma 26B (per-candidate, 1 step)", lambda n: _dgscore(n, 1)),
    "dgscore4":  ("DiffusionGemma 26B (per-candidate, 4 steps)", lambda n: _dgscore(n, 4)),
    "dgscore8":  ("DiffusionGemma 26B (per-candidate, 8 steps)", lambda n: _dgscore(n, 8)),
    "dgscore16": ("DiffusionGemma 26B (per-candidate, 16 steps)", lambda n: _dgscore(n, 16)),
    "dgmpll":    ("DiffusionGemma 26B (masked PLL)", lambda n: _dgmpll(n)),
    "ar7b":      ("Qwen2.5-7B (per-candidate log-likelihood)", lambda n: _ar(n)),
    "ar7bchoice":("Qwen2.5-7B (typed, local)", lambda n: _archoice(n)),
    "gpt2":      ("GPT-2 large (per-candidate log-likelihood)",
                  lambda n: _ar(n, "openai-community/gpt2-large")),
}


def _jev(n):
    from rescorers import JevRescorer
    return JevRescorer(max_options=n)

def _dg(n):
    from rescorers import DGChoiceRescorer
    return DGChoiceRescorer(max_options=n)

def _dgscore(n, k):
    from rescorers import DGScoreRescorer
    return DGScoreRescorer(max_options=n, steps=k)

def _dgmpll(n):
    from rescorers import DGMaskedPLLRescorer
    return DGMaskedPLLRescorer(max_options=n)

def _ar(n, model_id="mlx-community/Qwen2.5-7B-Instruct-4bit"):
    from rescorers import ARScoreRescorer
    return ARScoreRescorer(model_id=model_id, max_options=n)

def _archoice(n):
    from rescorers import ARChoiceRescorer
    return ARChoiceRescorer(max_options=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--limit", type=int, default=0, help="score only the first N sentences")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

    cache, meta = pickle.load(open(a.lists, "rb"))
    sents = list(cache)[: a.limit] if a.limit else list(cache)
    label, factory = ARMS[a.arm]
    scorer = factory(meta["nbest"])

    scores, conf, lat, ptok = {}, {}, {}, {}
    t_all = time.perf_counter()
    for i, s in enumerate(sents):
        hyps = [h for h, _, _ in cache[s]]
        t0 = time.perf_counter()
        sc = scorer(s, hyps)
        lat[s] = (time.perf_counter() - t0) * 1000
        scores[s] = [float(x) for x in sc]
        cs = getattr(scorer, "confidences", None)
        conf[s] = float(cs[-1]) if cs else None
        pt = getattr(scorer, "prompt_tokens", None)
        ptok[s] = int(pt) if pt else None
        if i % 25 == 0:
            print(f"  {i}/{len(sents)}  {np.median(list(lat.values())):.0f}ms median",
                  flush=True)

    out = a.out or f"scores/{os.path.basename(a.lists)[6:-4]}_{a.arm}.pkl"
    pickle.dump({"arm": a.arm, "label": label, "lists": a.lists, "list_meta": meta,
                 "scores": scores, "conf": conf, "latency": lat, "prompt_tokens": ptok,
                 "wall_s": time.perf_counter() - t_all}, open(out, "wb"))
    print(f"{label}: {len(sents)} sentences, median {np.median(list(lat.values())):.0f}ms "
          f"-> {out}")


if __name__ == "__main__":
    main()
