"""Format Willett et al. 2023 (T12) competitionData for phoneme decoding.

Follows the public baseline's notebook (github.com/cffan/neural_seq_decoder,
notebooks/formatCompetitionData.ipynb): area 6v tx1 + spike power (256 features), block-wise
z-scoring, g2p_en phonemes with a SIL after every word. One change, fixed in
PREREGISTRATION_T12.md: 10% of each session's `train` trials (seed 0) are held out as a
validation set for checkpoint selection, so the `test` partition is never used to pick a model.

    python t12_format.py --data ~/t12/data/competitionData --out ~/t12/t12_formatted.pkl
"""
import argparse, glob, os, pickle, re
import numpy as np, scipy.io
from g2p_en import G2p

PHONE_DEF = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
             'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
             'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
PHONE_DEF_SIL = PHONE_DEF + ['SIL']
g2p = G2p()


def load_session(path):
    dat = scipy.io.loadmat(path)
    feats, texts = [], []
    for i in range(dat['sentenceText'].shape[0]):
        feats.append(np.concatenate([dat['tx1'][0, i][:, 0:128], dat['spikePow'][0, i][:, 0:128]], axis=1)
                     .astype(np.float32))
        texts.append(str(dat['sentenceText'][i]).strip())
    blocks = np.squeeze(dat['blockIdx'])
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        allf = np.concatenate([feats[i] for i in idx], axis=0)
        mu, sd = allf.mean(0, keepdims=True), allf.std(0, keepdims=True)
        for i in idx:
            feats[i] = (feats[i] - mu) / (sd + 1e-8)
    return feats, texts


def phonemes(text):
    t = re.sub(r"[^a-zA-Z\- ']", '', text).replace('--', '').lower()
    out = []
    for p in g2p(t):
        if p == ' ':
            out.append('SIL')
        p = re.sub(r'[0-9]', '', p)
        if re.match(r'[A-Z]+', p):
            out.append(p)
    out.append('SIL')
    return np.array([PHONE_DEF_SIL.index(p) + 1 for p in out], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/t12/data/competitionData'))
    ap.add_argument('--out', default=os.path.expanduser('~/t12/t12_formatted.pkl'))
    ap.add_argument('--val-frac', type=float, default=0.10)
    a = ap.parse_args()
    sessions = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.data, 'train', '*.mat')))
    rng = np.random.default_rng(0)
    out = {'sessions': sessions, 'train': [], 'val': [], 'test': []}
    for s in sessions:
        f, t = load_session(os.path.join(a.data, 'train', s + '.mat'))
        perm = rng.permutation(len(f)); nval = int(round(a.val_frac * len(f)))
        val_idx, tr_idx = set(perm[:nval].tolist()), perm[nval:]
        for split, idx in (('train', sorted(tr_idx.tolist())), ('val', sorted(val_idx))):
            out[split].append({'feats': [f[i] for i in idx], 'text': [t[i] for i in idx],
                               'phon': [phonemes(t[i]) for i in idx]})
        tp = os.path.join(a.data, 'test', s + '.mat')
        if os.path.exists(tp):
            f2, t2 = load_session(tp)
            out['test'].append({'feats': f2, 'text': t2, 'phon': [phonemes(x) for x in t2]})
        else:
            out['test'].append({'feats': [], 'text': [], 'phon': []})
        print(s, {k: len(out[k][-1]['feats']) for k in ('train', 'val', 'test')}, flush=True)
    print('totals', {k: sum(len(d['feats']) for d in out[k]) for k in ('train', 'val', 'test')})
    pickle.dump(out, open(a.out, 'wb'))


if __name__ == '__main__':
    main()
