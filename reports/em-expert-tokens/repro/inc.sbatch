#!/bin/bash -l
#SBATCH -J inc
#SBATCH -o /ptmp/karthiksing/soft-moe/logs/%x_%j.out
#SBATCH -e /ptmp/karthiksing/soft-moe/logs/%x_%j.err
#SBATCH -D /u/karthiksing/projects/soft-moe
#SBATCH -A mpib_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=03:00:00
module purge; module load python-waterboa/2025.06
source /ptmp/karthiksing/soft-moe-venv/bin/activate
export HF_HOME=/ptmp/karthiksing/hf_cache PYTHONPATH=. HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e
M=Qwen/Qwen2.5-3B
D=/ptmp/karthiksing/soft-moe/persona_data_big
O=/ptmp/karthiksing/soft-moe/conv/inc; mkdir -p $O
BASE=$O/base_data; NEW=$O/new_data
TRB="python qwen_poc/train_sft.py --model $M --bs 8 --max-len 512"
EV(){ python qwen_poc/eval_poc.py --run "$1" --data $D --variant em --base-model $M 2>&1 | grep MACRO; }
echo "=== split: base = personas 0-6, new = persona 7 (robot) ==="
python qwen_poc/make_task_split.py --data $D --group-a 0,1,2,3,4,5,6 --out-a $BASE --out-b $NEW
echo "=== base backbone (Phase A on 7 base personas) ==="
$TRB --data $BASE --variant em --phase backbone --lr 1e-5 --steps 1000 --out $O/base_bb
echo "-- BASE_BB (step 0)"; EV $O/base_bb
echo "=== ARM 1: token-only (Phase B, backbone frozen) ==="
$TRB --data $NEW --variant em --phase tokens --lr 1e-2 --init-from $O/base_bb --steps 300 --save-steps 10,30,100 --out $O/tok
for s in 10 30 100; do echo "-- TOK s$s"; EV $O/tok__s$s; done; echo "-- TOK s300"; EV $O/tok
echo "=== ARM 2: full SFT (Phase full, backbone + token) ==="
$TRB --data $NEW --variant em --phase full --lr 1e-5 --init-from $O/base_bb --steps 300 --save-steps 10,30,100 --out $O/full
for s in 10 30 100; do echo "-- FULL s$s"; EV $O/full__s$s; done; echo "-- FULL s300"; EV $O/full
echo "=== ARM 3: EM two-phase (Phase A backbone 150 -> Phase B tokens 150) ==="
$TRB --data $NEW --variant em --phase backbone --lr 1e-5 --init-from $O/base_bb --steps 150 --save-steps 30,100 --out $O/emA
for s in 30 100; do echo "-- EMA s$s"; EV $O/emA__s$s; done; echo "-- EMA s150"; EV $O/emA
$TRB --data $NEW --variant em --phase tokens --lr 1e-2 --init-from $O/emA --steps 150 --save-steps 30,100 --out $O/emB
for s in 30 100; do echo "-- EMB s$s"; EV $O/emB__s$s; done; echo "-- EMB s150"; EV $O/emB
echo "=== data-efficiency: token-only, 100 steps, n new examples ==="
for n in 5 25 100; do
  Nn=$O/new_n$n; mkdir -p $Nn; cp $NEW/experts.json $NEW/control.test.jsonl $NEW/em.test.jsonl $Nn/
  shuf $NEW/em.train.jsonl | head -n $n > $Nn/em.train.jsonl
  shuf $NEW/control.train.jsonl | head -n $n > $Nn/control.train.jsonl
  $TRB --data $Nn --variant em --phase tokens --lr 1e-2 --init-from $O/base_bb --steps 100 --out $O/tokn$n
  echo "-- TOKN n$n"; EV $O/tokn$n
done
find $O -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
echo INC_DONE
