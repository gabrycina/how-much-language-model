#!/usr/bin/env python3
"""MWER discriminative fine-tuning of GPT-2 small for Kaldi n-best rescoring.

Protocol (from round-3 revision task):
  - Data: ~/workspace/jev-revision/data/lists_kaldi.pkl
  - Split: sorted list keys permuted with np.random.default_rng(0), first 30%
    = dev (419), rest = test (978). This matches the canonical round-3 pipeline
    (~/b2t/work/make_results.py::split on the L40S box); insertion order gives a
    different test set (9.389/8.968 instead of 8.091/7.983).
    Train = first 300 dev sentences, heldout = last 119 dev (early stopping).
    NEVER train on or tune to test.
  - Train: MWER loss on TOP-20 candidates/list, AdamW lr 1e-5, wd 0.01,
    batch = 4 lists, max 3 epochs, early stop on heldout expected-error.
  - Eval: per-candidate length-normalised log-likelihoods from the fine-tuned
    model on FULL dev+test lists; combination s(h) = (1-a)*z(decoder) +
    a*z(lm), decoder = 0.3*acoustic + ngram, z per-list; tune alpha on the
    FULL dev set, grid 0.00..1.00 step 0.05; report dev/test WER plus paired
    bootstrap p vs no-rescoring and vs cached untuned GPT-2 small.

Run on a GPU box (CPU on 2 cores is ~100x too slow; see round3_mwer_note.md).
"""
import os, pickle, math, random, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DATA = os.path.expanduser('~/workspace/jev-revision/data/lists_kaldi.pkl')
CACHED_GPT2SMALL = os.path.expanduser('~/workspace/jev-revision/data/kaldi_gpt2small.pkl')
OUTDIR = os.path.expanduser('~/workspace/jev-revision/analysis_out')
TOPK_TRAIN = 20
LR, WD, BATCH_LISTS, MAX_EPOCHS = 1e-5, 0.01, 4, 3
ALPHAS = [round(a * 0.05, 2) for a in range(21)]
BOOT_B, BOOT_SEED = 10000, 0

# ---------------------------------------------------------------- data ------
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

def load():
    lists_dict, meta = pickle.load(open(DATA, 'rb'))
    # Canonical round-3 split: sorted keys, np.random.default_rng(0) permutation
    # (matches ~/b2t/work/make_results.py::split). Insertion order is WRONG here.
    keys = sorted(lists_dict.keys())
    idx = np.random.default_rng(0).permutation(len(keys))
    n_dev = int(len(keys) * 0.30)                        # 419
    dev_keys = [keys[i] for i in idx[:n_dev]]
    test_keys = [keys[i] for i in idx[n_dev:]]
    train_keys, held_keys = dev_keys[:300], dev_keys[300:]
    def norm(k):                                        # strip surrounding quotes
        return k.strip().strip('"').strip("'")
    def pack(ks, topk=None):
        out = []
        for k in ks:
            cands = lists_dict[k][:topk] if topk else lists_dict[k]
            out.append({'ref': norm(k),
                        'text': [c[0] for c in cands],
                        'ac': [c[1] for c in cands],
                        'ng': [c[2] for c in cands]})
        return out
    return pack(train_keys, TOPK_TRAIN), pack(held_keys, TOPK_TRAIN), \
           pack(dev_keys), pack(test_keys)

# ------------------------------------------------- scoring helpers ----------
def batch_ll(model, tok, texts, device):
    """Length-normalised log-likelihood per candidate, batched with padding."""
    enc = tok(texts, padding=True, truncation=True, max_length=128,
              return_tensors='pt')
    ids, mask = enc['input_ids'].to(device), enc['attention_mask'].to(device)
    with torch.no_grad():
        logp = F.log_softmax(model(ids, attention_mask=mask).logits, -1)
    shifted = logp[:, :-1].gather(2, ids[:, 1:, None]).squeeze(-1)
    m = mask[:, 1:].float()
    tot = (shifted * m).sum(1)
    lens = m.sum(1).clamp_min(1)
    return (tot / lens).cpu()

def decoder_scores(lists):
    return [[0.3 * a + n for a, n in zip(L['ac'], L['ng'])] for L in lists]

def zscore_rows(rows):
    out = []
    for r in rows:
        r = np.asarray(r, dtype=float)
        sd = r.std()
        out.append((r - r.mean()) / sd if sd > 1e-12 else np.zeros_like(r))
    return out

def wers_at_alpha(lists, lm_rows, alpha):
    """WER/sent-acc of the alpha-interpolated system on a set of lists."""
    dec = zscore_rows(decoder_scores(lists))
    lm = zscore_rows(lm_rows)
    errs, refns, sent_ok = [], [], []
    for L, d, l in zip(lists, dec, lm):
        s = (1 - alpha) * np.asarray(d) + alpha * np.asarray(l)
        hyp = L['text'][int(np.argmax(s))]
        e = word_lev(hyp, L['ref'])
        errs.append(e); refns.append(max(1, len(L['ref'].split())))
        sent_ok.append(hyp == L['ref'])
    errs, refns = np.array(errs), np.array(refns)
    return 100 * errs.sum() / refns.sum(), 100 * np.mean(sent_ok), errs, refns

def tune_alpha(lists, lm_rows):
    best = min(((wers_at_alpha(lists, lm_rows, a)[0], a) for a in ALPHAS),
               key=lambda t: t[0])
    return best[1], best[0]

def boot_p(errs_new, refn, errs_base, B=BOOT_B, seed=BOOT_SEED):
    """One-sided paired bootstrap p for H0: new NOT better than base."""
    rng = np.random.default_rng(seed)
    n = len(errs_new)
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        w_new = 100 * errs_new[idx].sum() / refn[idx].sum()
        w_base = 100 * errs_base[idx].sum() / refn[idx].sum()
        deltas.append(w_new - w_base)
    deltas = np.array(deltas)
    return float((deltas >= 0).mean())

# ------------------------------------------------------------ training ------
def tok_cache(tok, lists):
    return [tok(L['text'], padding=True, truncation=True, max_length=128,
                return_tensors='pt') for L in lists]

def mwer_loss(model, enc, wer_targets, device):
    """Expected word-error under softmax(length-normed LLs); differentiable."""
    ids, mask = enc['input_ids'].to(device), enc['attention_mask'].to(device)
    logp = F.log_softmax(model(ids, attention_mask=mask).logits, -1)
    shifted = logp[:, :-1].gather(2, ids[:, 1:, None]).squeeze(-1)
    m = mask[:, 1:].float()
    ll = (shifted * m).sum(1) / m.sum(1).clamp_min(1)
    post = F.softmax(ll, 0)
    w = torch.tensor(wer_targets, device=device, dtype=ll.dtype)
    return (post * w).sum()

def train(model, tok, train_lists, held_lists, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    train_enc = tok_cache(tok, train_lists)
    train_wer = [[word_lev(t, L['ref']) / max(1, len(L['ref'].split()))
                  for t in L['text']] for L in train_lists]
    held_enc = tok_cache(tok, held_lists)
    held_wer = [[word_lev(t, L['ref']) / max(1, len(L['ref'].split()))
                 for t in L['text']] for L in held_lists]

    def held_loss():
        model.eval(); tot = 0.0
        with torch.no_grad():
            for enc, w in zip(held_enc, held_wer):
                tot += mwer_loss(model, enc, w, device).item()
        model.train(); return tot / len(held_enc)

    best, best_state, bad, epochs_run = held_loss(), None, 0, 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    order = list(range(len(train_lists)))
    for ep in range(MAX_EPOCHS):
        random.seed(ep); random.shuffle(order)
        for i in range(0, len(order), BATCH_LISTS):
            opt.zero_grad()
            loss, nb = 0.0, 0
            for j in order[i:i + BATCH_LISTS]:
                loss = loss + mwer_loss(model, train_enc[j], train_wer[j], device)
                nb += 1
            (loss / nb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        epochs_run = ep + 1
        hl = held_loss()
        print(f'epoch {epochs_run}: heldout expected-error {hl:.4f} (best {best:.4f})',
              flush=True)
        if hl < best - 1e-6:
            best, bad = hl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 1:                      # one non-improving epoch -> stop
                print('early stop'); break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return epochs_run, best

# ------------------------------------------------------------------ main ----
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, flush=True)
    train_lists, held_lists, dev_lists, test_lists = load()
    print(f'train {len(train_lists)} heldout {len(held_lists)} '
          f'dev {len(dev_lists)} test {len(test_lists)}', flush=True)

    tok = GPT2TokenizerFast.from_pretrained(os.path.join(OUTDIR, 'gpt2-local'))
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(os.path.join(OUTDIR, 'gpt2-local')).to(device)

    # sanity: untrained model pipeline must reproduce cached numbers
    cached = pickle.load(open(CACHED_GPT2SMALL, 'rb'))['scores']
    cache_rows = [cached[k.strip().strip('"').strip("'")] for k in
                  [L['ref'] for L in test_lists]]
    w, _, _, _ = wers_at_alpha(test_lists, cache_rows, 0.4)
    print(f'pipeline check: cached gpt2small @a=0.4 test WER {w:.3f} '
          f'(expect ~7.983)', flush=True)
    w0, _, _, _ = wers_at_alpha(test_lists,
                                [np.zeros(len(L['text'])) for L in test_lists], 0.0)
    print(f'pipeline check: no-rescoring test WER {w0:.3f} (expect ~8.091)',
          flush=True)

    epochs_run, best_held = train(model, tok, train_lists, held_lists, device)
    model.eval()
    torch.save(model.state_dict(), os.path.join(OUTDIR, 'round3_mwer_gpt2small.pt'))

    dev_lm = [batch_ll(model, tok, L['text'], device).tolist() for L in dev_lists]
    print('dev scored', flush=True)
    test_lm = [batch_ll(model, tok, L['text'], device).tolist() for L in test_lists]
    print('test scored', flush=True)
    pickle.dump({'arm': 'gpt2small-mwer', 'scores': dict(zip(
        [L['ref'] for L in dev_lists] + [L['ref'] for L in test_lists],
        dev_lm + test_lm))},
        open(os.path.join(OUTDIR, 'round3_mwer_scores.pkl'), 'wb'))

    alpha, dev_wer = tune_alpha(dev_lists, dev_lm)
    dw, dsa, _, _ = wers_at_alpha(dev_lists, dev_lm, alpha)
    tw, tsa, t_errs, t_refn = wers_at_alpha(test_lists, test_lm, alpha)

    _, _, n_errs, _ = wers_at_alpha(test_lists,
        [np.zeros(len(L['text'])) for L in test_lists], 0.0)
    _, _, c_errs, _ = wers_at_alpha(test_lists, cache_rows, 0.4)
    p_none = boot_p(t_errs, t_refn, n_errs)
    p_base = boot_p(t_errs, t_refn, c_errs)

    res = {'alpha': alpha, 'epochs_run': epochs_run,
           'best_heldout_expected_error': best_held,
           'dev_wer': dw, 'dev_sent_acc': dsa,
           'test_wer': tw, 'test_sent_acc': tsa,
           'p_vs_no_rescoring': p_none, 'p_vs_untuned_gpt2small': p_base}
    json.dump(res, open(os.path.join(OUTDIR, 'round3_mwer_results.json'), 'w'),
              indent=2)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
