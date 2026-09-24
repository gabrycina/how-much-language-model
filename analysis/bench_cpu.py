#!/usr/bin/env python3
"""Quick throughput benchmark for MWER training on this CPU."""
import os, pickle, time, sys
os.environ.setdefault('HF_HOME', os.path.expanduser('~/workspace/jev-revision/analysis_out/.hf-cache'))
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DATA = os.path.expanduser('~/workspace/jev-revision/data/lists_kaldi.pkl')
lists_dict, meta = pickle.load(open(DATA, 'rb'))
keys = list(lists_dict.keys())
n_dev = int(len(keys) * 0.30)
train_keys = keys[:300]

def word_lev(a, b):
    a, b = a.split(), b.split()
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[m]

LOCAL_MODEL = os.path.expanduser('~/workspace/jev-revision/analysis_out/gpt2-local')
tok = GPT2TokenizerFast.from_pretrained(LOCAL_MODEL)
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

def pack(ks, topk):
    out = []
    for k in ks:
        cands = lists_dict[k][:topk]
        out.append({'ref': k.strip().strip('"').strip("'"),
                    'text': [c[0] for c in cands]})
    return out

train_lists = pack(train_keys[:4], 20)
encs = [tok(L['text'], padding=True, truncation=True, max_length=128, return_tensors='pt')
        for L in train_lists]
wers = [[word_lev(t, L['ref']) / max(1, len(L['ref'].split())) for t in L['text']]
        for L in train_lists]

def mwer_step(enc, w):
    ids, mask = enc['input_ids'], enc['attention_mask']
    logp = F.log_softmax(model(ids, attention_mask=mask).logits, -1)
    shifted = logp[:, :-1].gather(2, ids[:, 1:, None]).squeeze(-1)
    m = mask[:, 1:].float()
    ll = (shifted * m).sum(1) / m.sum(1).clamp_min(1)
    post = F.softmax(ll, 0)
    w = torch.tensor(w, dtype=ll.dtype)
    return (post * w).sum()

# warmup then time 3 training steps (4 lists each, like the real batch)
mwer_step(encs[0], wers[0]).backward(); opt.zero_grad()
t0 = time.perf_counter()
for i in range(3):
    opt.zero_grad()
    loss = sum(mwer_step(e, w) for e, w in zip(encs, wers)) / 4
    loss.backward()
    opt.step()
t_step = (time.perf_counter() - t0) / 3
print(f'train step (4 lists x 20 cands, fwd+bwd): {t_step:.1f}s', flush=True)

# eval throughput: forward-only on full 100-cand lists
model.eval()
full_lists = pack(train_keys[:5], None)
t0 = time.perf_counter()
nseq = 0
with torch.no_grad():
    for L in full_lists:
        enc = tok(L['text'], padding=True, truncation=True, max_length=128, return_tensors='pt')
        ids, mask = enc['input_ids'], enc['attention_mask']
        logp = F.log_softmax(model(ids, attention_mask=mask).logits, -1)
        nseq += len(L['text'])
t_eval = time.perf_counter() - t0
print(f'eval: {nseq} seqs in {t_eval:.1f}s -> {nseq/t_eval:.1f} seq/s', flush=True)

# extrapolate
steps_per_epoch = 300 // 4
train_s = steps_per_epoch * t_step * 2.5   # ~2.5 epochs incl. early stop
held_s = 119 * 20 / (nseq / t_eval)
eval_s = 1397 * 100 / (nseq / t_eval)
total_h = (train_s + held_s + eval_s) / 3600
print(f'estimate: train {train_s/60:.0f}min + heldout {held_s/60:.0f}min + eval {eval_s/60:.0f}min = {total_h:.1f}h total', flush=True)
