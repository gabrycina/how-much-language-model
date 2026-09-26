"""Leak check: 83 test sentences also occur (as text) in the RNN's training split. Do the headline numbers hold without them?"""
import json, os, pickle, re, numpy as np, jiwer
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
cache, _ = pickle.load(open(f"{REPO}/data/lists_published.pkl", "rb"))
keys = sorted(cache); idx = np.random.default_rng(0).permutation(len(keys)); k = int(len(keys) * .3)
dev, test = [keys[i] for i in idx[:k]], [keys[i] for i in idx[k:]]
norm = lambda s: re.sub(r"[^a-z' ]", "", s.lower()).strip()
tr = set(norm(l) for l in open(f"{REPO}/data/train_sentences.txt"))
z = lambda v: (np.asarray(v, float) - np.mean(v)) / (np.std(v) + 1e-9)
def picks(sc, a, scale, ks):
    o = {}
    for s in ks:
        nb = cache[s]; f = scale * np.array([x for _, x, _ in nb]) + np.array([l for _, _, l in nb])
        t = f if sc is None or a == 0 else (1 - a) * z(f) + a * z(sc[s]); o[s] = nb[int(np.argmax(t))][0]
    return o
def e(r, h): m = jiwer.process_words([r], [h]); return m.substitutions + m.deletions + m.insertions
def wer(p, ks): return 100 * sum(e(s, p[s]) for s in ks) / sum(len(s.split()) for s in ks)
opt = pickle.load(open(f"{REPO}/data/scores_published/kaldi_opt.pkl", "rb"))["scores"]
seen = [s for s in test if norm(s) in tr]; unseen = [s for s in test if norm(s) not in tr]
out = {"n_seen": len(seen), "n_unseen": len(unseen)}
for name, ks in [("seen", seen), ("unseen", unseen), ("all", test)]:
    fp = picks(None, 0, .3, ks); po = picks(opt, .3, .3, ks); pj = picks(opt, .45, .4, ks)
    out[name] = {"fp": wer(fp, ks), "opt_paper": wer(po, ks), "opt_joint": wer(pj, ks)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlap_check.json"), "w"), indent=1)
