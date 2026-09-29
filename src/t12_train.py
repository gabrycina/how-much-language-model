"""Train the T12 phoneme decoder with the public baseline's architecture and hyperparameters.

Model: GRUDecoder from github.com/cffan/neural_seq_decoder (5-layer bidirectional GRU, 1024 units),
settings from its scripts/train_model.py. Unlike the baseline trainer, the checkpoint is selected
on a validation set held out from `train` (see t12_format.py); `test` is never read here.

    PYTHONPATH=~/t12/neural_seq_decoder/src python t12_train.py --data ~/t12/t12_formatted.pkl --out ~/t12/model
"""
import argparse, os, pickle, time
import numpy as np, torch
from edit_distance import SequenceMatcher
from neural_decoder.model import GRUDecoder

ARGS = dict(seqLen=150, maxTimeSeriesLen=1200, batchSize=64, lrStart=0.02, lrEnd=0.02, nUnits=1024,
            nBatch=10000, nLayers=5, seed=0, nClasses=40, nInputFeatures=256, dropout=0.4,
            whiteNoiseSD=0.8, constantOffsetSD=0.2, gaussianSmoothWidth=2.0, strideLen=4, kernelLen=32,
            bidirectional=True, l2_decay=1e-5)


def flatten(split):
    items = []
    for day, d in enumerate(split):
        for x, y in zip(d['feats'], d['phon']):
            items.append((x, y, day))
    return items


def batchify(items, device):
    xs = [torch.tensor(x) for x, _, _ in items]; ys = [torch.tensor(y) for _, y, _ in items]
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).to(device)
    Y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True).to(device)
    xl = torch.tensor([len(x) for x in xs], device=device); yl = torch.tensor([len(y) for y in ys], device=device)
    days = torch.tensor([d for _, _, d in items], device=device)
    return X, Y, xl, yl, days


def evaluate(model, items, device, loss_ctc, k, s):
    model.eval(); ed = tot = 0; losses = []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            X, Y, xl, yl, days = batchify(items[i:i + 64], device)
            pred = model(X, days); ol = ((xl - k) / s).to(torch.int32)
            losses.append(loss_ctc(pred.log_softmax(2).permute(1, 0, 2), Y, ol, yl).item())
            for j in range(pred.shape[0]):
                seq = torch.unique_consecutive(pred[j, :ol[j]].argmax(-1)).cpu().numpy()
                seq = [int(v) for v in seq if v != 0]; true = Y[j, :yl[j]].cpu().numpy().tolist()
                ed += SequenceMatcher(a=true, b=seq).distance(); tot += len(true)
    model.train()
    return float(np.mean(losses)), ed / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/t12/t12_formatted.pkl'))
    ap.add_argument('--out', default=os.path.expanduser('~/t12/model'))
    ap.add_argument('--nBatch', type=int, default=ARGS['nBatch'])
    a = ap.parse_args(); args = dict(ARGS, nBatch=a.nBatch)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(args['seed']); np.random.seed(args['seed']); device = 'cuda'
    data = pickle.load(open(a.data, 'rb'))
    train, val = flatten(data['train']), flatten(data['val'])
    print(f'train {len(train)} trials, val {len(val)} trials, {len(data["train"])} sessions', flush=True)
    model = GRUDecoder(neural_dim=args['nInputFeatures'], n_classes=args['nClasses'], hidden_dim=args['nUnits'],
                       layer_dim=args['nLayers'], nDays=len(data['train']), dropout=args['dropout'], device=device,
                       strideLen=args['strideLen'], kernelLen=args['kernelLen'],
                       gaussianSmoothWidth=args['gaussianSmoothWidth'], bidirectional=args['bidirectional']).to(device)
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args['lrStart'], betas=(0.9, 0.999), eps=0.1,
                           weight_decay=args['l2_decay'])
    rng = np.random.default_rng(args['seed']); best = 1e9; t0 = time.time(); log = []
    for b in range(args['nBatch']):
        X, Y, xl, yl, days = batchify([train[i] for i in rng.integers(0, len(train), args['batchSize'])], device)
        X = X + torch.randn(X.shape, device=device) * args['whiteNoiseSD']
        X = X + torch.randn([X.shape[0], 1, X.shape[2]], device=device) * args['constantOffsetSD']
        pred = model(X, days)
        loss = loss_ctc(pred.log_softmax(2).permute(1, 0, 2), Y,
                        ((xl - model.kernelLen) / model.strideLen).to(torch.int32), yl)
        opt.zero_grad(); loss.backward(); opt.step()
        if b % 100 == 0 or b == args['nBatch'] - 1:
            vl, vc = evaluate(model, val, device, loss_ctc, model.kernelLen, model.strideLen)
            log.append((b, vl, vc))
            if vc < best:
                best = vc; torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
            print(f'batch {b} val loss {vl:.3f} val PER {vc:.4f} best {best:.4f} ({time.time()-t0:.0f}s)', flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, 'final.pt'))
    pickle.dump({'args': args, 'nDays': len(data['train']), 'log': log}, open(os.path.join(a.out, 'train_log.pkl'), 'wb'))


if __name__ == '__main__':
    main()
