"""Run the trained T12 decoder over the `test` partition and save phoneme logits in the T15 format.

The output matches ~/b2t/work/real_logits.pkl: {'trials': [{'sentence', 'logits' (float16, [T, 41],
order BLANK, AA..ZH, SIL)}]}, so build_lists_kaldi.py runs on it unchanged.

    PYTHONPATH=~/t12/neural_seq_decoder/src python t12_logits.py --model ~/t12/model --out ~/t12/t12_logits.pkl
"""
import argparse, os, pickle
import numpy as np, torch
from neural_decoder.model import GRUDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/t12/t12_formatted.pkl'))
    ap.add_argument('--model', default=os.path.expanduser('~/t12/model'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default=os.path.expanduser('~/t12/t12_logits.pkl'))
    a = ap.parse_args()
    info = pickle.load(open(os.path.join(a.model, 'train_log.pkl'), 'rb')); args = info['args']
    model = GRUDecoder(neural_dim=args['nInputFeatures'], n_classes=args['nClasses'], hidden_dim=args['nUnits'],
                       layer_dim=args['nLayers'], nDays=info['nDays'], dropout=args['dropout'], device='cuda',
                       strideLen=args['strideLen'], kernelLen=args['kernelLen'],
                       gaussianSmoothWidth=args['gaussianSmoothWidth'], bidirectional=args['bidirectional']).cuda()
    model.load_state_dict(torch.load(os.path.join(a.model, 'best.pt'), map_location='cuda')); model.eval()
    data = pickle.load(open(a.data, 'rb')); out = []
    with torch.no_grad():
        for day, d in enumerate(data[a.split]):
            for i, (x, text) in enumerate(zip(d['feats'], d['text'])):
                X = torch.tensor(x)[None].cuda(); pred = model(X, torch.tensor([day]).cuda())
                n = int((x.shape[0] - model.kernelLen) / model.strideLen)
                out.append({'sentence': text, 'session': data['sessions'][day], 'trial': i,
                            'logits': pred[0, :n].float().cpu().numpy().astype(np.float16)})
    pickle.dump({'trials': out, 'split': a.split, 'model': 'best.pt'}, open(a.out, 'wb'))
    print(f'wrote {len(out)} trials -> {a.out}; logit dim {out[0]["logits"].shape}')


if __name__ == '__main__':
    main()
