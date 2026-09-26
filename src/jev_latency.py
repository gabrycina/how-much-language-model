"""Time Jev from the client side and split server time from transport.

Each of 80 random test lists is sent as a full request and paired with a two-option
request on the same warm keep-alive connection. The provider's proxy reports how long it
waited for the model service in `x-envoy-upstream-service-time`; end-to-end minus that is
transport. Also records billed input tokens per list. Needs TYPESAFE_API_KEY.

    python src/jev_latency.py            # writes analysis/jev_latency_run2.json
"""
import json, os, pickle, random, statistics as st, time
from revision_analysis import split
from rescorers import LABELS
from typed_api import Jev, ENDPOINT, MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, "..", ".env")):
    for line in open(os.path.join(HERE, "..", ".env")):
        if "=" in line:
            k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)

cache, _ = pickle.load(open(os.path.join(HERE, "..", "data", "lists_published.pkl"), "rb"))
_, test = split(cache)
random.seed(11); sample = random.sample(test, 80)
sess = Jev().session

def body(state, q):
    return json.dumps({"state": state, "model": MODEL, "questions": q})

def full(s):
    crit = {LABELS[i]: h for i, (h, _, _) in enumerate(cache[s][:100])}
    return body({"task": "A speech decoder produced these candidate sentences for what a person was "
                         "trying to say. Exactly one is what they meant.", "candidates": crit},
                {"sentence": {"type": "choice", "criteria": crit,
                              "instructions": "Which candidate is the sentence the person actually "
                                              "meant? Judge by which reads as fluent, natural English."}})

TINY = body({"task": "x", "candidates": {"A": "yes", "B": "no"}},
            {"sentence": {"type": "choice", "instructions": "Pick one.", "criteria": {"A": "yes", "B": "no"}}})

def call(b):
    t = time.perf_counter(); r = sess.post(ENDPOINT, data=b, timeout=20)
    e = (time.perf_counter() - t) * 1000
    r.raise_for_status()
    return e, float(r.headers.get("x-envoy-upstream-service-time", "nan")), r.json().get("usage", {}).get("input_tokens")

call(TINY); call(TINY)  # warm the connection
rows = []
for s in sample:
    ef, uf, tok = call(full(s)); et, ut, _ = call(TINY)
    rows.append({"n": len(cache[s][:100]), "e2e_full": ef, "srv_full": uf, "tok": tok, "e2e_tiny": et, "srv_tiny": ut})
summary = {k: {"median": st.median(r[k] for r in rows)} for k in ["e2e_full", "srv_full", "e2e_tiny", "srv_tiny", "tok"]}
summary["transport_full_median"] = st.median(r["e2e_full"] - r["srv_full"] for r in rows)
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "rows": rows}, open(os.path.join(HERE, "..", "analysis", "jev_latency_run2.json"), "w"), indent=1)
