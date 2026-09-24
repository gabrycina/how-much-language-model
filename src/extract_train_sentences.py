"""Collect the train-split sentences, which are the n-gram LM's only training data."""
import os
import sys

import pandas as pd
from omegaconf import OmegaConf

REPO = os.path.expanduser("~/b2t/repo")
sys.path.insert(0, os.path.join(REPO, "model_training"))
from evaluate_model_helpers import load_h5py_file  # noqa: E402

DATA = os.path.expanduser("~/b2t/data")
margs = OmegaConf.load(os.path.join(DATA, "t15_pretrained_rnn_baseline",
                                    "t15_pretrained_rnn_baseline", "checkpoint/args.yaml"))
csv = pd.read_csv(os.path.join(DATA, "t15_copyTaskData_description.csv"))
root = os.path.join(DATA, "t15_copyTask_neuralData", "hdf5_data_final")

out = []
for sess in sorted(margs["dataset"]["sessions"]):
    f = os.path.join(root, sess, "data_train.hdf5")
    if not os.path.exists(f):
        continue
    d = load_h5py_file(f, csv)
    out.extend(s.strip() for s in d["sentence_label"] if s and s.strip())

dst = os.path.expanduser("~/b2t/work/train_sentences.txt")
with open(dst, "w") as fh:
    fh.write("\n".join(out) + "\n")
words = {w for s in out for w in s.split()}
print(f"{len(out)} train sentences, {len(words)} unique words -> {dst}")
