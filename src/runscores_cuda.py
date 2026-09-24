"""Run one rescoring arm over the real n-best lists and save its raw scores."""
import argparse
import os
import pickle
import time

import numpy as np

MODELS = {
    "opt":  "facebook/opt-6.7b",
    "gpt2": "openai-community/gpt2-large",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gpt2medium": "openai-community/gpt2-medium",
    "gpt2small": "openai-community/gpt2",
    "distilgpt2": "distilbert/distilgpt2",
}

LABELS = {
    "opt":        "OPT-6.7b (per-candidate log-likelihood)",
    "optchoice":  "OPT-6.7b (typed selection)",
    "gpt2":       "GPT-2 large (per-candidate log-likelihood)",
    "gpt2choice": "GPT-2 large (typed selection)",
    "qwen":       "Qwen2.5-7B (per-candidate log-likelihood)",
    "qwenchoice": "Qwen2.5-7B (typed selection)",
    "gpt2medium": "GPT-2 medium (per-candidate log-likelihood)",
    "gpt2small":  "GPT-2 small (per-candidate log-likelihood)",
    "distilgpt2": "DistilGPT-2 (per-candidate log-likelihood)",
    "jev":        "Jev (typed selection, hosted)",
}


def build(arm, n):
    import rescorers_cuda as R

    if arm == "jev":
        return R.JevRescorer(max_options=n)
    base = arm.replace("choice", "")
    mid = MODELS[base]
    if arm.endswith("choice"):
        return R.ARChoiceRescorer(mid, max_options=n)
    return R.ARScoreRescorer(mid, max_options=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", default=os.path.expanduser("~/b2t/work/lists_real.pkl"))
    ap.add_argument("--arm", required=True, choices=list(LABELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

    cache, meta = pickle.load(open(a.lists, "rb"))
    sents = list(cache)[: a.limit] if a.limit else list(cache)
    scorer = build(a.arm, meta["nbest"])

    scores, conf, lat, ptok = {}, {}, {}, {}
    t0 = time.perf_counter()
    for i, s in enumerate(sents):
        hyps = [h for h, _, _ in cache[s]]
        t = time.perf_counter()
        sc = scorer(s, hyps)
        lat[s] = (time.perf_counter() - t) * 1000
        scores[s] = [float(x) for x in sc]
        cs = getattr(scorer, "confidences", None)
        conf[s] = float(cs[-1]) if cs else None
        pt = getattr(scorer, "prompt_tokens", None)
        ptok[s] = int(pt) if pt else None
        if i % 100 == 0:
            print(f"  {i}/{len(sents)}  {np.median(list(lat.values())):.0f}ms median",
                  flush=True)

    out = a.out or os.path.expanduser(f"~/b2t/work/scores/real_{a.arm}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pickle.dump({"arm": a.arm, "label": LABELS[a.arm], "lists": a.lists,
                 "list_meta": meta, "scores": scores, "conf": conf, "latency": lat,
                 "prompt_tokens": ptok, "wall_s": time.perf_counter() - t0},
                open(out, "wb"))
    print(f"{LABELS[a.arm]}: {len(sents)} sentences, "
          f"median {np.median(list(lat.values())):.0f}ms -> {out}")


if __name__ == "__main__":
    main()
