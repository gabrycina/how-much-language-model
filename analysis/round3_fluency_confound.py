#!/usr/bin/env python3
"""Round 3: prompt-confound (fluency bias) test for the typed Jev arm.

Cached-data only: no API calls, no inference.
- jev_pick: argmax over kaldi_jev.pkl 'scores' per-label log-probs
- fluency proxy: argmax over kaldi_gpt2.pkl length-normalized log-likelihoods
- truth: candidate whose text == reference (may be absent)
- test split: last 70% of 1397 in lists_dict insertion order (978 lists)
"""
import pickle, json
import numpy as np

DATA = "/home/hatch/workspace/jev-revision/data/"
OUT = "/home/hatch/workspace/jev-revision/analysis_out/"

lists_dict, meta = pickle.load(open(DATA + "lists_kaldi.pkl", "rb"))
jev_scores = pickle.load(open(DATA + "kaldi_jev.pkl", "rb"))["scores"]
gpt2_scores = pickle.load(open(DATA + "kaldi_gpt2.pkl", "rb"))["scores"]

refs = list(lists_dict.keys())
assert len(refs) == 1397, len(refs)
test_refs = refs[1397 - 978:]  # last 70% in insertion order
assert len(test_refs) == 978

def norm(s):
    # strip quotes per task note (no quoted keys found in practice)
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1]
    return s.strip()

rows = []
mismatches = {"jev_len": 0, "gpt2_len": 0, "missing_jev": 0, "missing_gpt2": 0}
for r in test_refs:
    cand_entries = lists_dict[r]
    texts = [c[0] for c in cand_entries]
    n = len(texts)
    if r not in jev_scores:
        mismatches["missing_jev"] += 1
        continue
    if r not in gpt2_scores:
        mismatches["missing_gpt2"] += 1
        continue
    js = np.array(jev_scores[r], dtype=float)
    gs = np.array(gpt2_scores[r], dtype=float)
    if len(js) != n:
        mismatches["jev_len"] += 1
        continue
    if len(gs) != n:
        mismatches["gpt2_len"] += 1
        continue
    jev_pick = int(np.argmax(js))
    fluency_argmax = int(np.argmax(gs))
    truth_idx = None
    rn = norm(r)
    for i, t in enumerate(texts):
        if norm(t) == rn:
            truth_idx = i
            break
    rows.append({
        "ref": r, "n": n, "jev_pick": jev_pick, "truth_idx": truth_idx,
        "fluency_argmax": fluency_argmax,
        "labels_used_js": int(np.sum(~np.isnan(js))),
    })

print("test lists:", len(test_refs), "rows kept:", len(rows), "mismatches:", mismatches)

res = {}
N = len(rows)
res["n_test"] = N

def frac(num, den):
    return None if den == 0 else num / den

# ---- Task 2: overall ----
truth_present = [x for x in rows if x["truth_idx"] is not None]
tp = sum(1 for x in truth_present if x["jev_pick"] == x["truth_idx"])
res["overall_truth_present_n"] = len(truth_present)
res["p_jev_pick_is_truth"] = frac(tp, len(truth_present))          # 67.9% check
res["p_jev_pick_is_fluency_argmax"] = frac(
    sum(1 for x in rows if x["jev_pick"] == x["fluency_argmax"]), N)
res["p_jev_pick_is_A"] = frac(sum(1 for x in rows if x["jev_pick"] == 0), N)

# labels used (sanity: 57 for Jev)
labels_used = sorted({x["jev_pick"] for x in rows})
res["labels_used_n"] = len(labels_used)
res["labels_used"] = labels_used

# ---- Task 3: confound test ----
dev = [x for x in rows if x["jev_pick"] != 0]
res["n_deviate_from_A"] = len(dev)
dev_truth_present = [x for x in dev if x["truth_idx"] is not None]
res["n_deviate_truth_present"] = len(dev_truth_present)
res["p_dev_pick_truth"] = frac(
    sum(1 for x in dev_truth_present if x["jev_pick"] == x["truth_idx"]),
    len(dev_truth_present))
res["p_dev_pick_fluency"] = frac(
    sum(1 for x in dev if x["jev_pick"] == x["fluency_argmax"]), len(dev))
res["p_dev_pick_truth_fluency_joint"] = frac(
    sum(1 for x in dev if x["truth_idx"] is not None
        and x["jev_pick"] == x["truth_idx"] == x["fluency_argmax"]),
    len(dev_truth_present))

# sharpest subset: truth and fluency disagree AND Jev deviates from A
sharp = [x for x in rows if x["truth_idx"] is not None
         and x["truth_idx"] != x["fluency_argmax"] and x["jev_pick"] != 0]
res["sharp_n"] = len(sharp)
res["sharp_p_jev_pick_truth"] = frac(
    sum(1 for x in sharp if x["jev_pick"] == x["truth_idx"]), len(sharp))
res["sharp_p_jev_pick_fluency"] = frac(
    sum(1 for x in sharp if x["jev_pick"] == x["fluency_argmax"]), len(sharp))
res["sharp_p_jev_pick_neither"] = frac(
    sum(1 for x in sharp if x["jev_pick"] not in (x["truth_idx"], x["fluency_argmax"])),
    len(sharp))
# within sharp, where did Jev pick A? excluded by construction, so record deviation composition
res["sharp_p_jev_pick_is_A"] = 0.0  # excluded by definition

# complement: truth and fluency agree, Jev deviates from A (where did it go?)
agree = [x for x in rows if x["truth_idx"] is not None
         and x["truth_idx"] == x["fluency_argmax"] and x["jev_pick"] != 0]
res["agree_disagree_n"] = len(agree)
res["agree_p_jev_pick_truth"] = frac(
    sum(1 for x in agree if x["jev_pick"] == x["truth_idx"]), len(agree))

# ---- Task 4: position/fluency entanglement at top ----
res["p_A_is_fluency_argmax"] = frac(
    sum(1 for x in rows if x["fluency_argmax"] == 0), N)
res["p_A_is_truth"] = frac(
    sum(1 for x in truth_present if x["truth_idx"] == 0), len(truth_present))

# Jev truth-selection rate when picking A vs deviating
pickA = [x for x in rows if x["jev_pick"] == 0]
pickA_tp = [x for x in pickA if x["truth_idx"] is not None]
res["n_pickA"] = len(pickA)
res["p_pickA_then_truth"] = frac(
    sum(1 for x in pickA_tp if x["truth_idx"] == 0), len(pickA_tp))
res["n_deviate"] = len(dev)
res["p_dev_then_truth"] = frac(
    sum(1 for x in dev_truth_present if x["jev_pick"] == x["truth_idx"]),
    len(dev_truth_present))

# extra: how often does the fluency argmax itself equal truth?
res["p_fluency_argmax_is_truth"] = frac(
    sum(1 for x in truth_present if x["fluency_argmax"] == x["truth_idx"]),
    len(truth_present))

# sanity for fluency proxy: P(jev_pick == fluency) among deviations where truth absent
dev_no_truth = [x for x in dev if x["truth_idx"] is None]
res["n_deviate_truth_absent"] = len(dev_no_truth)
res["p_dev_no_truth_pick_fluency"] = frac(
    sum(1 for x in dev_no_truth if x["jev_pick"] == x["fluency_argmax"]),
    len(dev_no_truth))

# Jev pick rank distribution (first few ranks)
from collections import Counter
rank_counts = Counter(x["jev_pick"] for x in rows)
res["jev_pick_rank_top10"] = {str(k): rank_counts[k] for k in range(10)}

with open(OUT + "round3_fluency_confound.json", "w") as f:
    json.dump(res, f, indent=2)

def pct(v):
    return "n/a" if v is None else f"{100*v:.1f}%"

lines = []
lines.append("FLUENCY-CONFOUND TEST (typed Jev arm), cached data only")
lines.append(f"Test lists analysed: {N} (last 70% of 1397, seed-0 split)")
lines.append("")
lines.append("OVERALL")
lines.append(f"- P(Jev pick == truth | truth in n-best, N={len(truth_present)}): "
             f"{pct(res['p_jev_pick_is_truth'])}  [paper check: 67.9%]")
lines.append(f"- P(Jev pick == fluency argmax, N={N}): {pct(res['p_jev_pick_is_fluency_argmax'])}")
lines.append(f"- P(Jev pick == A / label A mass, N={N}): {pct(res['p_jev_pick_is_A'])}")
lines.append(f"- Distinct labels used by Jev: {res['labels_used_n']}  [paper check: 57]")
lines.append(f"  labels: {res['labels_used']}")
lines.append("")
lines.append("CONFOUND TEST (Jev deviates from label A)")
lines.append(f"- Deviations from A: {res['n_deviate_from_A']} of {N} ({pct(res['n_deviate_from_A']/N)})")
lines.append(f"- Given deviation (and truth present, N={len(dev_truth_present)}): "
             f"picks truth {pct(res['p_dev_pick_truth'])}; "
             f"picks fluency argmax {pct(res['p_dev_pick_fluency'])}")
lines.append(f"- SHARP SUBSET: truth and fluency disagree AND Jev deviates "
             f"(N={res['sharp_n']}): picks truth {pct(res['sharp_p_jev_pick_truth'])}, "
             f"picks fluency argmax {pct(res['sharp_p_jev_pick_fluency'])}, "
             f"picks neither {pct(res['sharp_p_jev_pick_neither'])}")
lines.append(f"- CONTROL: truth == fluency argmax, Jev deviates (N={res['agree_disagree_n']}): "
             f"picks truth/fluent candidate {pct(res['agree_p_jev_pick_truth'])}")
lines.append(f"- Deviations where truth absent (N={res['n_deviate_truth_absent']}): "
             f"pick fluency argmax {pct(res['p_dev_no_truth_pick_fluency'])}")
lines.append("")
lines.append("POSITION/FLUENCY ENTANGLEMENT")
lines.append(f"- P(label A is the fluency argmax): {pct(res['p_A_is_fluency_argmax'])}")
lines.append(f"- P(label A is the truth | truth present): {pct(res['p_A_is_truth'])}")
lines.append(f"- Truth-selection rate when Jev picks A (N={res['n_pickA']}): "
             f"{pct(res['p_pickA_then_truth'])}")
lines.append(f"- Truth-selection rate when Jev deviates (N={len(dev_truth_present)}): "
             f"{pct(res['p_dev_then_truth'])}")
lines.append(f"- Fluency-argmax itself equals truth (oracle for fluency proxy): "
             f"{pct(res['p_fluency_argmax_is_truth'])}")
lines.append("")
lines.append("INTERPRETATION")
s = res['sharp_p_jev_pick_truth']; fl = res['sharp_p_jev_pick_fluency']
if s is not None and fl is not None:
    if s > fl:
        lines.append(f"- On the sharp subset Jev favours truth ({pct(s)}) over fluency ({pct(fl)}): "
                     "deviations are NOT driven by the prompt's fluency wording.")
    elif fl > s:
        lines.append(f"- On the sharp subset Jev favours the fluent candidate ({pct(fl)}) over truth ({pct(s)}): "
                     "evidence the prompt's fluency wording IS biasing selections.")
    else:
        lines.append("- Sharp subset shows no difference between truth and fluency.")
with open(OUT + "round3_fluency_confound.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
