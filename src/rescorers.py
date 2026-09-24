"""Interchangeable rescorers.

Each takes the n-best hypotheses and returns one score per hypothesis - the same
contract as `rescore_with_gpt2` in the Brain-to-Text baseline, so they drop into the
identical score combination.
"""

import math
import string

import numpy as np

LABELS = list(string.ascii_uppercase) + [f"{a}{b}" for a in string.ascii_uppercase
                                         for b in string.ascii_uppercase]


def no_rescore(context, hypotheses):
    return None


class JevRescorer:
    """Typed selection over the n-best list. The output cannot be a sentence that was
    not proposed, which in a device that speaks for someone is a safety property."""

    name = "jev"

    def __init__(self, max_options=100):
        from typed_api import Jev

        self.jev = Jev()
        self.max_options = max_options
        self.calls = 0
        self.confidences = []

    def __call__(self, context, hypotheses):
        hyps = hypotheses[: self.max_options]
        criteria = {LABELS[i]: h for i, h in enumerate(hyps)}
        answers, _ = self.jev.ask(
            {"task": "A speech decoder produced these candidate sentences for what a "
                     "person was trying to say. Exactly one is what they meant.",
             "candidates": criteria},
            {"sentence": {"type": "choice",
                          "instructions": "Which candidate is the sentence the person "
                                          "actually meant? Judge by which reads as "
                                          "fluent, natural English.",
                          "criteria": criteria}})
        self.calls += 1
        self.confidences.append(float(answers["sentence"].get("confidence", 0.0)))
        probs = answers["sentence"].get("probabilities", {})
        floor = 1e-6
        scores = [math.log(max(probs.get(LABELS[i], 0.0), floor)) for i in range(len(hyps))]
        scores += [math.log(floor)] * (len(hypotheses) - len(hyps))
        return scores


class MLXRescorer:
    """Local log-likelihood scoring with an MLX model on Apple silicon.

    This is the conventional way to rescore: score every hypothesis by how probable the
    language model thinks it is. One batched forward pass, no network.
    """

    name = "mlx-loglik"

    def __init__(self, model_id="mlx-community/diffusiongemma-26B-A4B-it-4bit"):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)
        self.model_id = model_id

    def __call__(self, context, hypotheses):
        import mlx.core as mx

        scores = []
        for h in hypotheses:
            ids = self.tokenizer.encode(h)
            if len(ids) < 2:
                scores.append(-1e9)
                continue
            x = mx.array([ids])
            logits = self.model(x[:, :-1])
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tgt = x[:, 1:]
            tok_lp = mx.take_along_axis(logprobs, tgt[..., None], axis=-1).squeeze(-1)
            scores.append(float(tok_lp.sum().item()) / len(ids))
        return scores


class MLXChoiceRescorer:
    """Typed selection on a local MLX model - the same shape as Jev, run on-device.

    Candidates are labelled with single tokens and listed in the prompt; one forward pass
    gives the logits at the answer slot, and the logprobs over the label tokens are the
    per-hypothesis scores. This mirrors vLLM PR #57250's read-only canvas trick, which
    fixes every slot but the answer and reads it in a single denoising step.

    Cost is one forward pass regardless of how many candidates there are, where scoring
    each hypothesis separately is linear in the list size.
    """

    name = "mlx-choice"

    def __init__(self, model_id="mlx-community/diffusiongemma-26B-A4B-it-4bit",
                 max_options=60):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)
        self.model_id = model_id
        self.max_options = max_options
        self.labels, self.label_ids = self._single_token_labels(max_options)

    def _single_token_labels(self, n):
        """Options must be one token each, or the answer slot shifts."""
        pool = ([f" {c}" for c in string.ascii_uppercase]
                + [f" {i}" for i in range(1, 200)]
                + [f" {c}" for c in string.ascii_lowercase])
        labels, ids = [], []
        for cand in pool:
            tok = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(tok) == 1 and tok[0] not in ids:
                labels.append(cand.strip())
                ids.append(tok[0])
            if len(labels) >= n:
                break
        return labels, ids

    def __call__(self, context, hypotheses):
        import mlx.core as mx

        hyps = hypotheses[: self.max_options]
        listing = "\n".join(f"{self.labels[i]}: {h}" for i, h in enumerate(hyps))
        prompt = (
            "A speech decoder produced these candidate sentences for what a person was "
            "trying to say. Exactly one is what they meant. Answer with the single label "
            "of the most fluent, natural English sentence.\n\n"
            f"{listing}\n\nAnswer:"
        )
        ids = self.tokenizer.encode(prompt)
        logits = self.model(mx.array([ids]))[0, -1]
        logprobs = logits - mx.logsumexp(logits)
        picks = np.array([float(logprobs[i].item()) for i in self.label_ids[: len(hyps)]])
        out = list(picks) + [float(picks.min()) - 10.0] * (len(hypotheses) - len(hyps))
        return out


class _DiffusionGemma:
    """Shared handle on the local DiffusionGemma checkpoint.

    Loading is ~5s and ~17GB resident, so every rescorer in a run shares one instance.
    """

    _cached = {}

    def __init__(self, model_id="mlx-community/diffusiongemma-26B-A4B-it-4bit"):
        from mlx_vlm import load

        if model_id not in _DiffusionGemma._cached:
            _DiffusionGemma._cached[model_id] = load(model_id)
        self.model, self.processor = _DiffusionGemma._cached[model_id]
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        self.vocab = self.model.config.text_config.vocab_size
        self.model_id = model_id

    def encode_prompt(self, text, answer_prefix=""):
        """`answer_prefix` opens the model's own turn, so the canvas slot that follows is
        exactly the label position rather than whatever the model would open a reply with."""
        import mlx.core as mx

        chat = self.tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return mx.array([self.tok.encode(chat + answer_prefix)])

    def seed_canvas(self, length, dtype, mode, key=None):
        """The canvas a denoising step starts from.

        The checkpoint's own sampler starts from uniform noise, which is fine when 48
        steps follow. A single step is sensitive to that noise, so `pad` and `fixed`
        hold the slot at a constant token instead.
        """
        import mlx.core as mx

        if mode == "random":
            return mx.random.randint(0, self.vocab, (1, length), key=key).astype(dtype)
        if mode == "pad":
            return mx.zeros((1, length), dtype=dtype)
        return mx.full((1, length), int(mode), dtype=dtype)

    def canvas_logits(self, input_ids, canvas):
        """One prefill over the prompt, one decoder pass over the canvas."""
        cache = self.model.diffusion_prefill_cache(input_ids)
        masks = self.model.diffusion_decoder_masks(canvas, cache, None)
        return self.model.diffusion_decoder_logits(
            canvas, cache=cache, self_conditioning=None, decoder_attention_mask=masks)


PROMPT = ("A speech decoder produced these candidate sentences for what a person was "
          "trying to say. Exactly one is what they meant. Answer with the single label "
          "of the most fluent, natural English sentence.\n\n{listing}\n\nAnswer:")


class DGChoiceRescorer:
    """Typed selection on DiffusionGemma, locally.

    The structured-generation mode of vLLM PR #57250 in miniature: the prompt fixes every
    token but one, a single denoising step fills the free slot, and the logits there -
    restricted to the label tokens - are the scores. One forward pass whatever the list
    length, and the answer is always one of the candidates.
    """

    name = "dg-choice"

    def __init__(self, model_id="mlx-community/diffusiongemma-26B-A4B-it-4bit",
                 max_options=60, seed_mode="pad", space=False, answer_prefix="The answer is "):
        self.dg = _DiffusionGemma(model_id)
        self.max_options = max_options
        self.seed_mode = seed_mode
        self.answer_prefix = answer_prefix
        self.labels, self.label_ids = self._single_token_labels(max_options, space)
        self.confidences = []

    def _single_token_labels(self, n, space):
        pre = " " if space else ""
        pool = ([f"{pre}{c}" for c in string.ascii_uppercase]
                + [f"{pre}{i}" for i in range(1, 300)]
                + [f"{pre}{c}" for c in string.ascii_lowercase])
        labels, ids = [], []
        for cand in pool:
            t = self.dg.tok.encode(cand, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                labels.append(cand.strip())
                ids.append(t[0])
            if len(labels) >= n:
                break
        if len(labels) < n:
            raise ValueError(f"only {len(labels)} single-token labels for {n} options")
        return labels, ids

    def __call__(self, context, hypotheses):
        import mlx.core as mx

        hyps = hypotheses[: self.max_options]
        listing = "\n".join(f"{self.labels[i]}: {h}" for i, h in enumerate(hyps))
        input_ids = self.dg.encode_prompt(PROMPT.format(listing=listing),
                                          self.answer_prefix)
        canvas = self.dg.seed_canvas(1, input_ids.dtype, self.seed_mode)
        logits = self.dg.canvas_logits(input_ids, canvas)[0, 0]
        lp = logits - mx.logsumexp(logits)
        picks = np.array([float(lp[i].item()) for i in self.label_ids[: len(hyps)]])
        norm = picks - np.log(np.exp(picks - picks.max()).sum()) - picks.max()
        self.confidences.append(float(np.exp(norm).max()))
        return list(picks) + [float(picks.min()) - 10.0] * (len(hypotheses) - len(hyps))


class DGLoglikRescorer:
    """The conventional rescoring shape, run on the same local model.

    Each hypothesis is written into the canvas and scored by the log-probability the
    model assigns to its own tokens in one denoising pass - the diffusion analogue of the
    pseudo-log-likelihood that OPT-6.7b supplies in the published baseline. It costs one
    forward pass *per hypothesis*, which is the whole point of the comparison.
    """

    name = "dg-loglik"

    def __init__(self, model_id="mlx-community/diffusiongemma-26B-A4B-it-4bit",
                 max_options=60):
        self.dg = _DiffusionGemma(model_id)
        self.max_options = max_options
        self.prefix = self.dg.encode_prompt(
            "Repeat the sentence a person was most likely trying to say.")

    def __call__(self, context, hypotheses):
        import mlx.core as mx

        # The instruction prefix is identical for every candidate, so it is encoded once
        # per list rather than once per candidate. Re-encoding it each time would charge
        # this arm for work the typed arm does not do, and make the cost comparison unfair.
        m = self.dg.model
        cache = m.diffusion_prefill_cache(self.prefix)
        scores = []
        for h in hypotheses[: self.max_options]:
            ids = self.dg.tok.encode(h, add_special_tokens=False)
            canvas = mx.array([ids]).astype(self.prefix.dtype)
            masks = m.diffusion_decoder_masks(canvas, cache, None)
            logits = m.diffusion_decoder_logits(canvas, cache=cache, self_conditioning=None,
                                                decoder_attention_mask=masks)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tok_lp = mx.take_along_axis(lp, canvas[..., None], axis=-1).squeeze(-1)
            scores.append(float(tok_lp.sum().item()) / max(len(ids), 1))
        return scores + [min(scores) - 10.0] * (len(hypotheses) - len(scores))


class DGScoreRescorer:
    """Per-candidate scoring on the diffusion model with K refinement passes.

    The first draft used a single pass, which a reviewer correctly called underpowered:
    diffusion LMs are normally queried with tens of denoising steps. Here the canvas is
    held at the candidate (we are scoring, not generating) and K passes are run, each
    feeding the previous pass's logits back as self-conditioning, exactly as the
    checkpoint's own sampler does. The candidate's token log-probabilities are read off
    the final pass.
    """

    name = "dg-score"

    def __init__(self, model_id="mlx-community/diffusiongemma-26B-A4B-it-4bit",
                 max_options=60, steps=1):
        self.dg = _DiffusionGemma(model_id)
        self.max_options = max_options
        self.steps = steps
        self.prefix = self.dg.encode_prompt(
            "Repeat the sentence a person was most likely trying to say.")
        self.sc_ctx = self.dg.model.diffusion_prepare_self_conditioning()
        self.prompt_tokens = int(self.prefix.shape[1])

    def _final_logits(self, canvas, cache):
        import mlx.core as mx

        m = self.dg.model
        masks = m.diffusion_decoder_masks(canvas, cache, None)
        sc, logits = None, None
        for _ in range(self.steps):
            logits = m.diffusion_decoder_logits(canvas, cache=cache, self_conditioning=sc,
                                                decoder_attention_mask=masks)
            sc = m.diffusion_self_conditioning(logits, self.sc_ctx)
            mx.eval(sc)
        return logits

    def __call__(self, context, hypotheses):
        import mlx.core as mx

        # Prefix encoded once per list, not once per candidate - see DGLoglikRescorer.
        cache = self.dg.model.diffusion_prefill_cache(self.prefix)
        scores = []
        for h in hypotheses[: self.max_options]:
            ids = self.dg.tok.encode(h, add_special_tokens=False)
            canvas = mx.array([ids]).astype(self.prefix.dtype)
            logits = self._final_logits(canvas, cache)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tok_lp = mx.take_along_axis(lp, canvas[..., None], axis=-1).squeeze(-1)
            scores.append(float(tok_lp.sum().item()) / max(len(ids), 1))
        return scores + [min(scores) - 10.0] * (len(hypotheses) - len(scores))


class DGMaskedPLLRescorer:
    """Masked pseudo-log-likelihood on the diffusion model, after Salazar et al. (2020).

    A token cannot be scored fairly while the model can see it. Positions are masked in
    `stride` interleaved groups, so each token is predicted with itself hidden, at a cost
    of `stride` passes per candidate rather than one per token.
    """

    name = "dg-mpll"

    def __init__(self, model_id="mlx-community/diffusiongemma-26B-A4B-it-4bit",
                 max_options=60, stride=4, mask_token=0):
        self.dg = _DiffusionGemma(model_id)
        self.max_options = max_options
        self.stride = stride
        self.mask_token = mask_token
        self.prefix = self.dg.encode_prompt(
            "Repeat the sentence a person was most likely trying to say.")
        self.prompt_tokens = int(self.prefix.shape[1])

    def __call__(self, context, hypotheses):
        import mlx.core as mx

        m = self.dg.model
        cache = m.diffusion_prefill_cache(self.prefix)
        scores = []
        for h in hypotheses[: self.max_options]:
            ids = self.dg.tok.encode(h, add_special_tokens=False)
            clean = mx.array([ids]).astype(self.prefix.dtype)
            masks = m.diffusion_decoder_masks(clean, cache, None)
            total = 0.0
            for off in range(min(self.stride, len(ids))):
                pos = list(range(off, len(ids), self.stride))
                cur = [self.mask_token if i in set(pos) else t for i, t in enumerate(ids)]
                canvas = mx.array([cur]).astype(self.prefix.dtype)
                logits = m.diffusion_decoder_logits(canvas, cache=cache,
                                                    self_conditioning=None,
                                                    decoder_attention_mask=masks)
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                for i in pos:
                    total += float(lp[0, i, ids[i]].item())
            scores.append(total / max(len(ids), 1))
        return scores + [min(scores) - 10.0] * (len(hypotheses) - len(scores))


class _MLXLM:
    """Shared handle on an autoregressive MLX checkpoint."""

    _cached = {}

    def __init__(self, model_id):
        from mlx_lm import load

        if model_id not in _MLXLM._cached:
            _MLXLM._cached[model_id] = load(model_id)
        self.model, self.tokenizer = _MLXLM._cached[model_id]
        self.model_id = model_id


class ARScoreRescorer:
    """The published rescoring shape: an autoregressive LM's sequence log-likelihood.

    This is the control the first draft lacked. No MLX port of OPT-6.7b exists - mlx-lm
    carries no `opt` architecture - so the same-scale stand-in is used instead, which if
    anything makes the control stronger rather than weaker.
    """

    name = "ar-score"

    def __init__(self, model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
                 max_options=60, length_norm=True):
        self.lm = _MLXLM(model_id)
        self.max_options = max_options
        self.length_norm = length_norm
        self.model_id = model_id

    def __call__(self, context, hypotheses):
        import mlx.core as mx

        tok = self.lm.tokenizer
        scores = []
        for h in hypotheses[: self.max_options]:
            ids = tok.encode(h)
            if len(ids) < 2:
                scores.append(-1e9)
                continue
            x = mx.array([ids])
            logits = self.lm.model(x[:, :-1])
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tgt = x[:, 1:]
            tok_lp = mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
            total = float(tok_lp.sum().item())
            scores.append(total / (len(ids) - 1) if self.length_norm else total)
        return scores + [min(scores) - 10.0] * (len(hypotheses) - len(scores))


class ARChoiceRescorer:
    """Typed selection on an autoregressive model - the fourth cell of the 2x2.

    Same prompt and same label alphabet as the diffusion typed arm; the difference is only
    that the answer slot is the next-token position rather than a canvas slot.
    """

    name = "ar-choice"

    def __init__(self, model_id="mlx-community/Qwen2.5-7B-Instruct-4bit", max_options=60):
        self.lm = _MLXLM(model_id)
        self.max_options = max_options
        self.model_id = model_id
        self.labels, self.label_ids = self._labels(max_options)
        self.confidences = []
        self.prompt_tokens = 0

    def _labels(self, n):
        tok = self.lm.tokenizer
        pool = ([f"{c}" for c in string.ascii_uppercase]
                + [f"{i}" for i in range(1, 400)]
                + [f"{c}" for c in string.ascii_lowercase])
        labels, ids = [], []
        for cand in pool:
            t = tok.encode(cand, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                labels.append(cand)
                ids.append(t[0])
            if len(labels) >= n:
                break
        if len(labels) < n:
            raise ValueError(f"only {len(labels)} single-token labels for {n} options")
        return labels, ids

    def __call__(self, context, hypotheses):
        import mlx.core as mx

        tok = self.lm.tokenizer
        hyps = hypotheses[: self.max_options]
        listing = "\n".join(f"{self.labels[i]}: {h}" for i, h in enumerate(hyps))
        chat = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(listing=listing)}],
            add_generation_prompt=True, tokenize=False)
        ids = tok.encode(chat + "The answer is ")
        self.prompt_tokens = len(ids)
        logits = self.lm.model(mx.array([ids]))[0, -1]
        lp = logits - mx.logsumexp(logits)
        picks = np.array([float(lp[i].item()) for i in self.label_ids[: len(hyps)]])
        norm = picks - picks.max()
        self.confidences.append(float(np.exp(norm).max() / np.exp(norm).sum()))
        return list(picks) + [float(picks.min()) - 10.0] * (len(hypotheses) - len(hyps))
