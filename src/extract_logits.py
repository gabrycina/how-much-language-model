"""Run the pretrained RNN over the real neural recordings to get phoneme logits.

This replaces the simulated front end entirely. Everything downstream - the beam search,
the n-best lists, the rescoring comparison - now consumes the output of a real decoder
reading a real intracortical array.

Only the `val` split is usable: the `test` split ships without ground-truth sentences, so
word error rate cannot be computed on it.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO = os.path.expanduser("~/b2t/repo")
sys.path.insert(0, os.path.join(REPO, "model_training"))
from evaluate_model_helpers import (LOGIT_TO_PHONEME, load_h5py_file,  # noqa: E402
                                    runSingleDecodingStep)
from rnn_model import GRUDecoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/b2t/data"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=os.path.expanduser("~/b2t/work/real_logits.pkl"))
    ap.add_argument("--limit-sessions", type=int, default=0)
    a = ap.parse_args()

    model_path = os.path.join(a.data, "t15_pretrained_rnn_baseline",
                              "t15_pretrained_rnn_baseline")
    margs = OmegaConf.load(os.path.join(model_path, "checkpoint/args.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUDecoder(
        neural_dim=margs["model"]["n_input_features"],
        n_units=margs["model"]["n_units"],
        n_days=len(margs["dataset"]["sessions"]),
        n_classes=margs["dataset"]["n_classes"],
        rnn_dropout=margs["model"]["rnn_dropout"],
        input_dropout=margs["model"]["input_network"]["input_layer_dropout"],
        n_layers=margs["model"]["n_layers"],
        patch_size=margs["model"]["patch_size"],
        patch_stride=margs["model"]["patch_stride"],
    )
    ckpt = torch.load(os.path.join(model_path, "checkpoint/best_checkpoint"),
                      weights_only=False, map_location=device)
    # The checkpoint is saved from a compiled/DDP wrapper; strip those prefixes.
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
          for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"loaded RNN: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params "
          f"on {device}", flush=True)

    csv = pd.read_csv(os.path.join(a.data, "t15_copyTaskData_description.csv"))
    root = os.path.join(a.data, "t15_copyTask_neuralData", "hdf5_data_final")
    sessions = sorted(margs["dataset"]["sessions"])
    if a.limit_sessions:
        sessions = sessions[: a.limit_sessions]

    out, t0, n = [], time.perf_counter(), 0
    for si, sess in enumerate(sessions):
        f = os.path.join(root, sess, f"data_{a.split}.hdf5")
        if not os.path.exists(f):
            continue
        data = load_h5py_file(f, csv)
        day_idx = margs["dataset"]["sessions"].index(sess)
        for t in range(len(data["neural_features"])):
            x = torch.tensor(data["neural_features"][t], device=device,
                             dtype=torch.bfloat16).unsqueeze(0)
            logits = runSingleDecodingStep(x, day_idx, model, margs, device)
            out.append({"session": sess, "trial": t,
                        "sentence": data["sentence_label"][t],
                        "logits": logits[0].astype(np.float16)})
            n += 1
        print(f"  [{si+1}/{len(sessions)}] {sess}: {len(data['neural_features'])} trials "
              f"({n} total, {time.perf_counter()-t0:.0f}s)", flush=True)

    pickle.dump({"trials": out, "phonemes": LOGIT_TO_PHONEME, "split": a.split},
                open(a.out, "wb"))
    print(f"\nwrote {len(out)} trials of real phoneme logits -> {a.out}")
    print(f"logit dim: {out[0]['logits'].shape}, phoneme set: {len(LOGIT_TO_PHONEME)}")


if __name__ == "__main__":
    main()
