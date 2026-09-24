"""A small phoneme-to-words beam decoder - the n-gram stage of a speech BCI.

Given the decoder's noisy phoneme read, search the vocabulary for word sequences that
explain it, scored by phonetic match plus a bigram language model. This is what produces
the n-best list, and crucially the true sentence has to *compete* for its place in it:
generating candidates from the truth instead would leak the answer and make rescoring
look pointless.
"""

from collections import defaultdict

import numpy as np

from nbest import phoneme_neighbours


class BeamDecoder:
    def __init__(self, builder, beam=400, lm_weight=0.3, ins_penalty=0.0):
        self.b = builder
        self.beam = beam
        self.lm_weight = lm_weight
        self.ins_penalty = ins_penalty
        # Index words by first phoneme, including the ones it is confusable with, so a
        # corrupted first sound does not hide the right word.
        self.by_first = defaultdict(set)
        for w, ph in builder.pron.items():
            if not ph:
                continue
            self.by_first[ph[0]].add(w)
            for alt in phoneme_neighbours(ph[0]):
                self.by_first[alt].add(w)

    def match_cost(self, word_ph, obs, i):
        """Cost of explaining obs[i:] with this word, and how many phonemes it consumes."""
        best = (1e9, len(word_ph))
        for span in (len(word_ph) - 1, len(word_ph), len(word_ph) + 1):
            if span <= 0 or i + span > len(obs):
                continue
            c = self.b.seq_distance(word_ph, obs[i:i + span])
            if c < best[0]:
                best = (c, span)
        return best

    def decode(self, obs, n=100):
        """Return [(hypothesis, acoustic_score, lm_score)] ranked by the decoder itself."""
        # beams: position -> list of (score, words, acoustic, lm)
        beams = {0: [(0.0, (), 0.0, 0.0)]}
        finished = []
        cost_cache = {}
        for i in range(len(obs)):
            if i not in beams:
                continue
            pool = sorted(beams.pop(i), key=lambda t: -t[0])[: self.beam]
            for score, words, ac, lm in pool:
                prev = words[-1] if words else "<s>"
                for w in self.by_first.get(obs[i], ()):
                    ph = self.b.pron.get(w)
                    if not ph:
                        continue
                    key = (w, i)
                    if key not in cost_cache:
                        cost_cache[key] = self.match_cost(ph, obs, i)
                    cost, span = cost_cache[key]
                    if cost > len(ph) * 0.9 + 1.0:      # too poor an explanation
                        continue
                    nxt = i + span
                    new_ac = ac - cost - self.ins_penalty
                    new_lm = lm + self.b.word_logp(prev, w)
                    new_score = new_ac + self.lm_weight * new_lm
                    entry = (new_score, words + (w,), new_ac, new_lm)
                    if nxt >= len(obs) - 1:
                        finished.append(entry)
                    else:
                        beams.setdefault(nxt, []).append(entry)

        if not finished:
            return []
        finished.sort(key=lambda t: -t[0])
        out, seen = [], set()
        for _, words, ac, _lm in finished:
            h = " ".join(words)
            if h in seen:
                continue
            seen.add(h)
            out.append((h, float(ac), self.b.lm_score(list(words))))
            if len(out) >= n:
                break
        return out
