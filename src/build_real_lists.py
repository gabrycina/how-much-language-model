"""Turn real RNN phoneme logits into n-best candidate lists.

A lexicon-constrained CTC beam search (flashlight, via torchaudio) with a KenLM n-gram
takes the place of the published Kaldi/WFST stage: the pretrained 3-gram and 5-gram models
are distributed separately and are not retrievable, so the n-gram is trained here on the
train split, which is disjoint from the val split everything is evaluated on.

Output format matches the simulated pipeline exactly - [(text, acoustic, lm)] per sentence -
so every downstream rescoring arm runs unchanged.
"""
import argparse
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

_PUNCT = str.maketrans("", "", ",.?!-\u2019'\";:()[]")


def norm(t):
    """Word error rate in this field is computed on lowercased, unpunctuated text.
    References carry case and punctuation that the lexicon cannot emit, so comparing
    raw strings scores a perfect decode as an error."""
    return " ".join(t.translate(_PUNCT).lower().split())

PHON = ['BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
        'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
        'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', '|']


def build_lexicon(words, path):
    """word -> phoneme spelling, from CMUdict, in flashlight lexicon format."""
    from nltk.corpus import cmudict
    import re
    cmu = cmudict.dict()
    n = 0
    with open(path, "w") as f:
        for w in sorted(words):
            prons = cmu.get(w)
            if not prons:
                continue
            seen = set()
            for p in prons[:2]:                       # at most two pronunciations
                toks = [re.sub(r"\d", "", x) for x in p]
                if any(t not in PHON for t in toks):
                    continue
                key = tuple(toks)
                if key in seen:
                    continue
                seen.add(key)
                f.write(f"{w}\t{' '.join(toks)} |\n")
                n += 1
    return n


def train_lm(sentences, order, out_arpa):
    txt = out_arpa + ".txt"
    with open(txt, "w") as f:
        for s in sentences:
            f.write(s.strip() + "\n")
    subprocess.run(["lmplz", "-o", str(order), "--discount_fallback",
                    "--text", txt, "--arpa", out_arpa],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_arpa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default=os.path.expanduser("~/b2t/work/real_logits.pkl"))
    ap.add_argument("--train-sentences", default=os.path.expanduser("~/b2t/work/train_sentences.txt"))
    ap.add_argument("--nbest", type=int, default=60)
    ap.add_argument("--beam", type=int, default=150)
    ap.add_argument("--lm-weight", type=float, default=2.0)
    ap.add_argument("--word-score", type=float, default=0.0)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/b2t/work/lists_real.pkl"))
    a = ap.parse_args()

    from torchaudio.models.decoder import ctc_decoder

    d = pickle.load(open(a.logits, "rb"))
    trials = d["trials"][: a.limit] if a.limit else d["trials"]
    print(f"{len(trials)} real trials", flush=True)

    train_sents = [l.strip() for l in open(a.train_sentences) if l.strip()]
    print(f"LM corpus: {len(train_sents)} train-split sentences (disjoint from val)",
          flush=True)

    # Lexicon and LM live in the same normalised space as the references.
    train_sents = [norm(s) for s in train_sents]
    vocab = sorted({w for s in train_sents for w in s.split()}
                   | {w for t in trials for w in norm(t["sentence"]).split()})
    lex_path = os.path.expanduser("~/b2t/work/lexicon.txt")
    n_lex = build_lexicon(vocab, lex_path)
    print(f"lexicon: {n_lex} pronunciations over {len(vocab)} words", flush=True)

    arpa = train_lm(train_sents, a.order, os.path.expanduser("~/b2t/work/lm.arpa"))
    print(f"trained {a.order}-gram KenLM", flush=True)

    dec = ctc_decoder(
        lexicon=lex_path, tokens=PHON, lm=arpa,
        nbest=a.nbest, beam_size=a.beam, beam_size_token=30,
        lm_weight=a.lm_weight, word_score=a.word_score,
        blank_token="BLANK", sil_token="|", unk_word="<unk>",
    )

    cache, oracle, inlist, t0 = {}, [], [], time.perf_counter()
    import jiwer
    for i, tr in enumerate(trials):
        lg = torch.tensor(tr["logits"].astype(np.float32)).unsqueeze(0)
        lp = torch.log_softmax(lg, dim=-1)
        hyps = dec(lp)[0]
        nb = []
        for h in hyps:
            text = " ".join(h.words).strip()
            text = norm(text)
            if text:
                nb.append((text, 0.0, float(h.score)))
        if not nb:
            continue
        # Deduplicate while preserving the decoder's own ranking.
        seen, ded = set(), []
        for t_, ac, lm in nb:
            if t_ not in seen:
                seen.add(t_); ded.append((t_, ac, lm))
        ref = norm(tr["sentence"])
        cache[ref] = ded
        oracle.append(min(jiwer.wer(ref, h) for h, _, _ in ded))
        inlist.append(ref in [h for h, _, _ in ded])
        if i % 100 == 0:
            print(f"  {i}/{len(trials)}  oracle {np.mean(oracle)*100:.1f}%  "
                  f"in-list {np.mean(inlist)*100:.0f}%  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    meta = {"regime": "real", "source": "T15 intracortical, pretrained RNN, CTC beam+KenLM",
            "nbest": a.nbest, "beam": a.beam, "lm_order": a.order,
            "lm_weight": a.lm_weight, "n": len(cache),
            "oracle_wer": float(np.mean(oracle)) * 100,
            "truth_in_nbest": float(np.mean(inlist)) * 100,
            "mean_list_len": float(np.mean([len(v) for v in cache.values()])),
            "lexicon": len(vocab), "lm_train": len(train_sents)}
    pickle.dump((cache, meta), open(a.out, "wb"))
    print(f"\n{meta['n']} sentences | oracle {meta['oracle_wer']:.1f}% | "
          f"truth in list {meta['truth_in_nbest']:.0f}% | "
          f"mean |list| {meta['mean_list_len']:.0f} -> {a.out}")


if __name__ == "__main__":
    main()
