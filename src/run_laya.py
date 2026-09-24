"""Score the real n-best lists with Laya, an open 421M typed-decision model, on-device.

Same interface as the hosted arm - state plus a typed `choice` question - so the only
thing that differs from the Jev arm is which model answers.
"""
import pickle
import string
import time

import numpy as np
from laya import Router

LABELS = list(string.ascii_uppercase) + [f"{a}{b}" for a in string.ascii_uppercase
                                         for b in string.ascii_uppercase]

import sys
LISTS = sys.argv[1] if len(sys.argv) > 1 else "real/lists_real.pkl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "real/scores/real_laya.pkl"
cache, meta = pickle.load(open(LISTS, "rb"))
router = Router(preload=True)

scores, conf, lat, ptok = {}, {}, {}, {}
t0 = time.perf_counter()
for i, ref in enumerate(cache):
    hyps = [h for h, _, _ in cache[ref]][: meta["nbest"]]
    crit = {LABELS[j]: h for j, h in enumerate(hyps)}
    qs = {"sentence": {"type": "choice",
                       "instructions": "Which candidate is the sentence the person "
                                       "actually meant? Judge by which reads as fluent, "
                                       "natural English.",
                       "criteria": crit}}
    state = {"task": "A speech decoder produced these candidate sentences for what a "
                     "person was trying to say. Exactly one is what they meant.",
             "candidates": crit}
    full = [h for h, _, _ in cache[ref]]
    if len(hyps) < 2:
        # Nothing to choose: a one-candidate list is decided before any rescorer sees it.
        scores[ref] = [0.0] * len(full)
        conf[ref] = 1.0
        lat[ref] = 0.0
        continue
    t = time.perf_counter()
    r = router.predict(state, qs)
    lat[ref] = (time.perf_counter() - t) * 1000
    a = r["answers"]["sentence"]
    probs = a.get("probabilities", {}) or {}
    floor = 1e-6
    sc = [float(np.log(max(probs.get(LABELS[j], 0.0), floor))) for j in range(len(hyps))]
    scores[ref] = sc + [float(np.log(floor))] * (len(full) - len(sc))
    conf[ref] = float(a.get("confidence", 0.0))
    if i % 200 == 0:
        print(f"  {i}/{len(cache)}  {np.median(list(lat.values())):.0f}ms median", flush=True)

pickle.dump({"arm": "laya", "label": "Laya 421M (typed selection, on-device)",
             "lists": LISTS, "list_meta": meta, "scores": scores,
             "conf": conf, "latency": lat, "prompt_tokens": ptok,
             "wall_s": time.perf_counter() - t0},
            open(OUT, "wb"))
print(f"Laya: {len(scores)} sentences, median {np.median(list(lat.values())):.0f}ms "
      f"-> {OUT}")
