"""Rescoring arms on CUDA.

Two shapes are compared on every model: per-candidate scoring (the published shape, one
sequence log-likelihood per candidate) and typed selection (one constrained forward pass
per list). Per-candidate scoring is *batched*, because an unbatched baseline would flatter
the typed arm on cost for reasons that have nothing to do with the question being asked.
"""
import math
import string

import numpy as np
import torch

LABELS = list(string.ascii_uppercase) + [f"{a}{b}" for a in string.ascii_uppercase
                                         for b in string.ascii_uppercase]

PROMPT = ("A speech decoder produced these candidate sentences for what a person was "
          "trying to say. Exactly one is what they meant. Answer with the single label "
          "of the most fluent, natural English sentence.\n\n{listing}\n\nAnswer:")


def no_rescore(context, hypotheses):
    return None


class _HF:
    """One shared copy of each checkpoint, in bf16 on the GPU."""

    _cache = {}

    def __init__(self, model_id, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_id not in _HF._cache:
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            m = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map="cuda", low_cpu_mem_usage=True)
            m.eval()
            _HF._cache[model_id] = (m, tok)
        self.model, self.tok = _HF._cache[model_id]
        self.model_id = model_id


class ARScoreRescorer:
    """Per-candidate sequence log-likelihood - the shape the published baseline uses."""

    name = "ar-score"

    def __init__(self, model_id, max_options=100, length_norm=True, batch_size=32):
        self.lm = _HF(model_id)
        self.max_options = max_options
        self.length_norm = length_norm
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, context, hypotheses):
        tok, model = self.lm.tok, self.lm.model
        hyps = hypotheses[: self.max_options]
        scores = []
        for i in range(0, len(hyps), self.batch_size):
            chunk = hyps[i: i + self.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
            ids, mask = enc["input_ids"], enc["attention_mask"]
            out = model(input_ids=ids, attention_mask=mask).logits.float()
            lp = torch.log_softmax(out[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            tokl = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            valid = mask[:, 1:].float()
            tot = (tokl * valid).sum(1)
            n = valid.sum(1).clamp(min=1)
            scores.extend((tot / n if self.length_norm else tot).tolist())
        pad = [min(scores) - 10.0] * (len(hypotheses) - len(scores)) if scores else []
        return scores + pad


class ARChoiceRescorer:
    """Typed selection: the whole list in one prompt, answer constrained to a label."""

    name = "ar-choice"

    def __init__(self, model_id, max_options=100, scaffold="The answer is "):
        self.lm = _HF(model_id)
        self.max_options = max_options
        self.scaffold = scaffold
        self.labels, self.label_ids = self._labels(max_options)
        self.confidences = []
        self.prompt_tokens = 0

    def _labels(self, n):
        tok = self.lm.tok
        pool = (list(string.ascii_uppercase) + [str(i) for i in range(1, 400)]
                + list(string.ascii_lowercase)
                + [a + b for a in string.ascii_uppercase
                   for b in string.ascii_uppercase])
        labels, ids = [], []
        for c in pool:
            t = tok.encode(c, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                labels.append(c)
                ids.append(t[0])
            if len(labels) >= n:
                break
        if len(labels) < n:
            raise ValueError(f"only {len(labels)} single-token labels for {n} options")
        return labels, ids

    @torch.no_grad()
    def __call__(self, context, hypotheses):
        tok, model = self.lm.tok, self.lm.model
        hyps = hypotheses[: self.max_options]
        listing = "\n".join(f"{self.labels[i]}: {h}" for i, h in enumerate(hyps))
        text = PROMPT.format(listing=listing)
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, tokenize=False)
        ids = tok(text + self.scaffold, return_tensors="pt").to("cuda")
        self.prompt_tokens = int(ids["input_ids"].shape[1])
        logits = model(**ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        picks = np.array([float(lp[i]) for i in self.label_ids[: len(hyps)]])
        e = np.exp(picks - picks.max())
        self.confidences.append(float((e / e.sum()).max()))
        return list(picks) + [float(picks.min()) - 10.0] * (len(hypotheses) - len(hyps))


class JevRescorer:
    """Typed selection through the hosted constrained-decoding API."""

    name = "jev"

    def __init__(self, max_options=100):
        from typed_api import Jev

        self.jev = Jev()
        self.max_options = max_options
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
        self.confidences.append(float(answers["sentence"].get("confidence", 0.0)))
        probs = answers["sentence"].get("probabilities", {})
        floor = 1e-6
        sc = [math.log(max(probs.get(LABELS[i], 0.0), floor)) for i in range(len(hyps))]
        return sc + [math.log(floor)] * (len(hypotheses) - len(hyps))
