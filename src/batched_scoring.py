"""Does batching rescue per-candidate scoring?

Per-candidate scoring runs the model once per candidate. A competent implementation would
batch those calls, so comparing an unbatched baseline against a single-pass typed read
overstates the cost gap. Batching changes *cost*, not scores, so this measures latency and
verifies the scores are unchanged rather than re-running the whole benchmark.
"""
import json
import pickle
import time

import mlx.core as mx
import numpy as np

SAMPLE = 8


def ar_unbatched(lm, hyps):
    tok = lm.tokenizer
    out = []
    for h in hyps:
        ids = tok.encode(h)
        x = mx.array([ids])
        logits = lm.model(x[:, :-1])
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        t = mx.take_along_axis(lp, x[:, 1:][..., None], axis=-1).squeeze(-1)
        out.append(float(t.sum().item()) / (len(ids) - 1))
    return out


def ar_batched(lm, hyps, pad_id=0):
    """One padded forward pass over the whole list; padding masked out of the sum."""
    tok = lm.tokenizer
    seqs = [tok.encode(h) for h in hyps]
    L = max(len(s) for s in seqs)
    x = mx.array([s + [pad_id] * (L - len(s)) for s in seqs])
    logits = lm.model(x[:, :-1])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = x[:, 1:]
    tokl = mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
    valid = mx.array([[1.0] * (len(s) - 1) + [0.0] * (L - len(s)) for s in seqs])
    tot = (tokl * valid).sum(axis=1)
    n = mx.array([float(len(s) - 1) for s in seqs])
    return [float(v) for v in (tot / n).tolist()]


def dg_unbatched(dg, prefix, hyps):
    m = dg.model
    cache = m.diffusion_prefill_cache(prefix)
    out = []
    for h in hyps:
        ids = dg.tok.encode(h, add_special_tokens=False)
        canvas = mx.array([ids]).astype(prefix.dtype)
        masks = m.diffusion_decoder_masks(canvas, cache, None)
        logits = m.diffusion_decoder_logits(canvas, cache=cache, self_conditioning=None,
                                            decoder_attention_mask=masks)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        t = mx.take_along_axis(lp, canvas[..., None], axis=-1).squeeze(-1)
        out.append(float(t.sum().item()) / max(len(ids), 1))
    return out


def dg_batched_by_length(dg, prefix, hyps):
    """Group candidates of identical token length so no padding enters the canvas.

    The decoder attends bidirectionally across the canvas, so padding a batch would let
    real positions attend to pad tokens. Same-length grouping sidesteps that entirely.
    """
    m = dg.model
    cache = m.diffusion_prefill_cache(prefix)
    enc = [dg.tok.encode(h, add_special_tokens=False) for h in hyps]
    groups = {}
    for i, ids in enumerate(enc):
        groups.setdefault(len(ids), []).append(i)
    out = [0.0] * len(hyps)
    for L, idxs in groups.items():
        canvas = mx.array([enc[i] for i in idxs]).astype(prefix.dtype)
        masks = m.diffusion_decoder_masks(canvas, cache, None)
        logits = m.diffusion_decoder_logits(canvas, cache=cache, self_conditioning=None,
                                            decoder_attention_mask=masks)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        t = mx.take_along_axis(lp, canvas[..., None], axis=-1).squeeze(-1)
        s = t.sum(axis=1) / max(L, 1)
        for j, i in enumerate(idxs):
            out[i] = float(s[j].item())
    return out


def timed(fn, reps=3):
    fn()                                  # warm the graph
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), r


def main():
    from rescorers import _DiffusionGemma, _MLXLM

    cache, meta = pickle.load(open("lists_easy_s0.pkl", "rb"))
    sents = [s for s in cache if len(cache[s]) >= 60][:SAMPLE]
    report = {}

    lm = _MLXLM("mlx-community/Qwen2.5-7B-Instruct-4bit")
    ub, bb, agree = [], [], []
    for s in sents:
        hyps = [h for h, _, _ in cache[s]][:60]
        t_u, r_u = timed(lambda: ar_unbatched(lm, hyps))
        t_b, r_b = timed(lambda: ar_batched(lm, hyps))
        ub.append(t_u); bb.append(t_b)
        agree.append(float(np.max(np.abs(np.array(r_u) - np.array(r_b)))))
    report["qwen7b"] = {"unbatched_ms": float(np.median(ub)),
                        "batched_ms": float(np.median(bb)),
                        "speedup": float(np.median(ub) / np.median(bb)),
                        "max_score_diff": float(np.max(agree))}
    print(f"Qwen-7B per-candidate: unbatched {np.median(ub):.0f}ms -> batched "
          f"{np.median(bb):.0f}ms  ({np.median(ub)/np.median(bb):.1f}x)  "
          f"max |score diff| {np.max(agree):.2e}", flush=True)

    dg = _DiffusionGemma()
    prefix = dg.encode_prompt("Repeat the sentence a person was most likely trying to say.")
    ub, bb, agree, err = [], [], [], None
    for s in sents:
        hyps = [h for h, _, _ in cache[s]][:60]
        try:
            t_u, r_u = timed(lambda: dg_unbatched(dg, prefix, hyps))
            t_b, r_b = timed(lambda: dg_batched_by_length(dg, prefix, hyps))
            ub.append(t_u); bb.append(t_b)
            agree.append(float(np.max(np.abs(np.array(r_u) - np.array(r_b)))))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
    if err:
        report["diffusiongemma"] = {"error": err}
        print(f"DiffusionGemma batched: unavailable - {err[:110]}")
    else:
        report["diffusiongemma"] = {"unbatched_ms": float(np.median(ub)),
                                    "batched_ms": float(np.median(bb)),
                                    "speedup": float(np.median(ub) / np.median(bb)),
                                    "max_score_diff": float(np.max(agree))}
        print(f"DiffusionGemma per-candidate: unbatched {np.median(ub):.0f}ms -> batched "
              f"{np.median(bb):.0f}ms  ({np.median(ub)/np.median(bb):.1f}x)  "
              f"max |score diff| {np.max(agree):.2e}", flush=True)

    report["n_sentences"] = len(sents)
    report["peak_gb"] = float(mx.get_peak_memory() / 1e9)
    json.dump(report, open("batched_scoring.json", "w"), indent=2)
    print("wrote batched_scoring.json")


if __name__ == "__main__":
    main()
