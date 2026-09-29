#!/bin/bash
# T12: decoder outputs -> Kaldi candidate lists -> every rescoring arm. Same code and options as T15.
set -u
until [ -f ~/t12/model/final.pt ]; do sleep 30; done
cd ~/t12 && PYTHONPATH=~/t12/neural_seq_decoder/src .venv/bin/python t12_logits.py > logits.log 2>&1 || { echo LOGITS_FAIL; exit 1; }
cd ~/b2t/work && ~/b2t/.venv39/bin/python build_lists_kaldi.py --logits ~/t12/t12_logits.pkl --out ~/t12/lists_t12.pkl > ~/t12/lists.log 2>&1 || { echo LISTS_FAIL; exit 1; }
source ~/b2t/.venv/bin/activate; export PYTHONPATH=.; mkdir -p ~/t12/scores
( python runscores_cuda.py --lists ~/t12/lists_t12.pkl --arm jev --out ~/t12/scores/t12_jev.pkl > ~/t12/arm_jev.log 2>&1; echo JEV_DONE >> ~/t12/arm_jev.log ) &
for arm in opt qwen optchoice qwenchoice; do
  python runscores_cuda.py --lists ~/t12/lists_t12.pkl --arm $arm --out ~/t12/scores/t12_$arm.pkl 2>&1 | grep -vE "it/s\]|Loading weights|Fetching" > ~/t12/arm_$arm.log
done
python run_laya.py ~/t12/lists_t12.pkl ~/t12/scores/t12_laya.pkl > ~/t12/arm_laya.log 2>&1
wait
echo T12_ALL_DONE
