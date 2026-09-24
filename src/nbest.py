"""Build n-best hypothesis lists that behave like a speech-BCI decoder's output.

A speech neuroprosthesis emits phoneme probabilities, and an n-gram decoder turns those
into a ranked list of candidate sentences. The correct sentence is often *not* rank 1 -
that gap is exactly what LLM rescoring recovers (23.8% -> 17.4% WER in Willett et al.).

We reproduce that structure: confusable-phoneme substitutions produce candidates, each
scored by phonetic distance (standing in for the acoustic score) and a bigram language
model (standing in for the n-gram pass). Noise is calibrated so the pre-rescoring WER
lands near the published figure.
"""

import random
import re
from collections import defaultdict

import numpy as np

# Phoneme confusion groups: sounds a decoder actually mixes up (place/manner neighbours).
CONFUSABLE = [
    set("P B".split()), set("T D".split()), set("K G".split()),
    set("F V".split()), set("S Z".split()), set("SH ZH".split()),
    set("TH DH".split()), set("M N NG".split()), set("L R".split()),
    set("IH IY".split()), set("EH AE".split()), set("AA AO".split()),
    set("UH UW".split()), set("AH AX".split()), set("CH JH".split()),
]


def phoneme_neighbours(p):
    base = re.sub(r"\d", "", p)
    for group in CONFUSABLE:
        if base in group:
            return group - {base}
    return set()


class NBestBuilder:
    def __init__(self, lm_sentences, lexicon_sentences=None, seed=0, extra_words=0):
        """`lm_sentences` trains the bigram model; `lexicon_sentences` only supplies the
        word list. They must differ from the test set, or the language model has already
        memorised the answers and every rescorer looks perfect."""
        from g2p_en import G2p

        self.g2p = G2p()
        self.rng = random.Random(seed)
        sentences = lm_sentences
        self.sentences = sentences

        lex_src = lexicon_sentences if lexicon_sentences is not None else lm_sentences
        self.vocab = sorted({w for s in lex_src for w in s.split()})
        self.pron = {w: [re.sub(r"\d", "", p) for p in self.g2p(w) if p != " "]
                     for w in self.vocab}

        # A dictionary the size of the corpus makes the decoder's job unrealistically easy:
        # there are few words for a misheard sound to be confused with. Padding it out of
        # CMUdict - whose pronunciations are used directly, so this stays cheap - is how the
        # hard regime is built.
        if extra_words:
            from nltk.corpus import cmudict

            cmu = cmudict.dict()
            pool = sorted(w for w in cmu
                          if w.isalpha() and 2 <= len(w) <= 12 and w not in self.pron)
            rng = random.Random(seed + 7717)
            for w in rng.sample(pool, min(extra_words, len(pool))):
                self.pron[w] = [re.sub(r"\d", "", p) for p in cmu[w][0]]
            self.vocab = sorted(self.pron)

        # Words that sound alike are the ones a decoder swaps.
        self.by_shape = defaultdict(list)
        for w, ph in self.pron.items():
            self.by_shape[(len(ph), ph[0] if ph else "")].append(w)

        self.bigrams = defaultdict(lambda: defaultdict(int))
        self.unigrams = defaultdict(int)
        for s in sentences:
            toks = ["<s>"] + s.split()
            for a, b in zip(toks, toks[1:]):
                self.bigrams[a][b] += 1
                self.unigrams[b] += 1
        self.total = sum(self.unigrams.values())

    def phon_distance(self, a, b):
        pa, pb = self.pron.get(a, []), self.pron.get(b, [])
        if not pa or not pb:
            return 6.0
        # Cheap alignment: length gap plus per-slot mismatch, discounted when the
        # mismatch is between sounds a decoder confuses anyway.
        d = abs(len(pa) - len(pb)) * 1.0
        for x, y in zip(pa, pb):
            if x == y:
                continue
            d += 0.35 if y in phoneme_neighbours(x) else 1.0
        return d

    def confusions(self, word, k=4):
        ph = self.pron.get(word, [])
        if not ph:
            return []
        pool = self.by_shape[(len(ph), ph[0])] + self.by_shape[(len(ph), "")]
        cands = [w for w in set(pool) if w != word]
        cands.sort(key=lambda w: self.phon_distance(word, w))
        return cands[:k]

    def word_logp(self, prev, w):
        """One word's bigram log-probability. Kept separate so the decoder can extend a
        hypothesis in O(1) instead of rescoring the whole sentence at every step."""
        num = self.bigrams[prev].get(w, 0)
        den = sum(self.bigrams[prev].values())
        p_bi = num / den if den else 0.0
        p_uni = (self.unigrams.get(w, 0) + 1) / (self.total + len(self.vocab))
        return float(np.log(0.7 * p_bi + 0.3 * p_uni + 1e-12))

    def lm_score(self, words):
        score, prev = 0.0, "<s>"
        for w in words:
            num = self.bigrams[prev].get(w, 0)
            den = sum(self.bigrams[prev].values())
            p_bi = num / den if den else 0.0
            p_uni = (self.unigrams.get(w, 0) + 1) / (self.total + len(self.vocab))
            score += np.log(0.7 * p_bi + 0.3 * p_uni + 1e-12)
            prev = w
        return float(score)

    def observe(self, words, error_rate):
        """What the neural decoder actually emits: a corrupted phoneme sequence.

        This is the crucial bit. Scoring candidates against the *truth* leaks the answer
        and makes the task trivial. A real decoder only ever sees its own noisy read, so
        candidates are scored against that - and the correct sentence is frequently not
        the closest match, which is precisely why rescoring is worth anything.
        """
        obs = []
        for w in words:
            for p in self.pron.get(w, []):
                if self.rng.random() < error_rate:
                    alts = phoneme_neighbours(p)
                    obs.append(self.rng.choice(sorted(alts)) if alts else p)
                else:
                    obs.append(p)
        return obs

    def seq_distance(self, a, b):
        """Levenshtein over phonemes, discounted for confusable pairs."""
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                sub = 0.0 if x == y else (0.35 if y in phoneme_neighbours(x) else 1.0)
                cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + sub))
            prev = cur
        return prev[-1]

    def build(self, sentence, n=100, sub_rate=0.30, error_rate=0.22):
        """Return [(hypothesis, acoustic_score, lm_score)] scored against a noisy read."""
        truth = sentence.split()
        observed = self.observe(truth, error_rate)
        seen, out = set(), []

        def add(words):
            h = " ".join(words)
            if h in seen:
                return
            seen.add(h)
            phones = [p for w in words for p in self.pron.get(w, [])]
            acoustic = -self.seq_distance(phones, observed)
            out.append((h, acoustic, self.lm_score(words)))

        add(truth)
        guard = 0
        while len(out) < n and guard < n * 25:
            guard += 1
            words = list(truth)
            for i, w in enumerate(words):
                if self.rng.random() < sub_rate:
                    alts = self.confusions(w)
                    if alts:
                        words[i] = self.rng.choice(alts)
            add(words)
        return out[:n]
