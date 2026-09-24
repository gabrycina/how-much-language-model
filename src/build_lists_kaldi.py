"""Decode the real RNN logits with the published Kaldi/WFST decoder and OpenWebText 3-gram.

This replaces our own CTC beam search and 8k-sentence 3-gram with the ones the published
system actually uses, so the first pass is no longer a weaker stand-in. It also returns
separate acoustic and language-model scores per hypothesis, which lets the rescoring
combination use the published three-term form rather than a two-term approximation.

Decoder parameters are the published defaults from language_model/README.md:
    nbest 100, acoustic_scale 0.325, blank_penalty 90, max_active 7000, min_active 200.
"""
import argparse
import os
import pickle
import time

import numpy as np

import lm_decoder


def rearrange(logits):
    """The RNN emits [BLANK, phonemes..., SIL]; the decoder expects [BLANK, SIL, phonemes...]."""
    return np.concatenate([logits[:, 0:1], logits[:, -1:], logits[:, 1:-1]], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default=os.path.expanduser("~/b2t/data/languageModel"))
    ap.add_argument("--logits", default=os.path.expanduser("~/b2t/work/real_logits.pkl"))
    ap.add_argument("--nbest", type=int, default=100)
    ap.add_argument("--acoustic-scale", type=float, default=0.3)
    ap.add_argument("--ctc-blank-skip-threshold", type=float, default=1.0)
    ap.add_argument("--length-penalty", type=float, default=0.0)
    ap.add_argument("--blank-penalty", type=float, default=90.0)
    ap.add_argument("--max-active", type=int, default=7000)
    ap.add_argument("--min-active", type=int, default=200)
    ap.add_argument("--beam", type=float, default=17.0)
    ap.add_argument("--lattice-beam", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/b2t/work/lists_kaldi.pkl"))
    a = ap.parse_args()

    _PUNCT = str.maketrans("", "", ",.?!-’'\";:()[]")

    def norm(t):
        return " ".join(t.translate(_PUNCT).lower().split())

    print("loading TLG.fst (~15 GB, this takes a few minutes)...", flush=True)
    t0 = time.perf_counter()
    # Order is (max_active, min_active, beam, lattice_beam, acoustic_scale,
    # ctc_blank_skip_threshold, length_penalty, nbest). A zero blank-skip threshold
    # silences the decoder completely; the shipped default is 1.0.
    opts = lm_decoder.DecodeOptions(a.max_active, a.min_active, a.beam, a.lattice_beam,
                                    a.acoustic_scale, a.ctc_blank_skip_threshold,
                                    a.length_penalty, a.nbest)
    res = lm_decoder.DecodeResource(os.path.join(a.lm, "TLG.fst"), "", "",
                                    os.path.join(a.lm, "words.txt"), "")
    dec = lm_decoder.BrainSpeechDecoder(res, opts)
    print(f"  loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    d = pickle.load(open(a.logits, "rb"))
    trials = d["trials"][: a.limit] if a.limit else d["trials"]

    import jiwer
    cache, oracle, inlist, t0 = {}, [], [], time.perf_counter()
    for i, tr in enumerate(trials):
        lg = rearrange(tr["logits"].astype(np.float32))
        dec.Reset()
        lm_decoder.DecodeNumpy(dec, lg, np.zeros_like(lg), np.log(a.blank_penalty))
        dec.FinishDecoding()
        out = dec.result()
        nb, seen = [], set()
        for h in out:
            text = norm(h.sentence)
            if text and text not in seen:
                seen.add(text)
                nb.append((text, float(h.ac_score), float(h.lm_score)))
        if not nb:
            continue
        ref = norm(tr["sentence"])
        cache[ref] = nb
        oracle.append(min(jiwer.wer(ref, h) for h, _, _ in nb))
        inlist.append(ref in [h for h, _, _ in nb])
        if i % 100 == 0:
            print(f"  {i}/{len(trials)}  oracle {np.mean(oracle)*100:.1f}%  "
                  f"in-list {np.mean(inlist)*100:.0f}%  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    meta = {"regime": "real-kaldi",
            "source": "T15 intracortical, pretrained RNN, published Kaldi WFST decoder, "
                      "OpenWebText 3-gram",
            "nbest": a.nbest, "acoustic_scale": a.acoustic_scale,
            "blank_penalty": a.blank_penalty, "n": len(cache),
            "oracle_wer": float(np.mean(oracle)) * 100,
            "truth_in_nbest": float(np.mean(inlist)) * 100,
            "mean_list_len": float(np.mean([len(v) for v in cache.values()]))}
    pickle.dump((cache, meta), open(a.out, "wb"))
    print(f"\n{meta['n']} sentences | oracle {meta['oracle_wer']:.1f}% | "
          f"truth in list {meta['truth_in_nbest']:.0f}% | "
          f"mean |list| {meta['mean_list_len']:.0f} -> {a.out}")


if __name__ == "__main__":
    main()
