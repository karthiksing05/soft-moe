#!/bin/bash -l
#SBATCH -J syninc
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
# Incremental training with IMPOSSIBLE synthetic transforms: train a backbone on 7 transforms, then add an
# 8th GENUINELY NOVEL transform (len_prefix) three ways. The acid test vs the style version: the new
# capability is NOT latent in the backbone, so does a single frozen-backbone token embedding (token-only)
# suffice, or does adding a novel computation require backbone updates (full-SFT / EM two-phase)?
M=Qwen/Qwen2.5-3B
D=/ptmp/karthiksing/soft-moe/syninc_data
O=/ptmp/karthiksing/soft-moe/conv/syninc; mkdir -p $O
BASE=$O/base_data; NEW=$O/new_data
TRB="python qwen_poc/train_sft.py --model $M --bs 8 --max-len 128"
NEWACC(){ python qwen_poc/syntax_demo.py --run "$1" --data $D --variant em --only 7             --show 0 --max-eval 480 2>&1 | grep OVERALL | sed 's/^/    new(len_prefix) /'; }
BASEACC(){ python qwen_poc/syntax_demo.py --run "$1" --data $D --variant em --only 0,1,2,3,4,5,6 --show 0 --max-eval 480 2>&1 | grep OVERALL | sed 's/^/    base(0-6)       /'; }
echo "=== generate synthetic-syntax data (K=8) ==="
python qwen_poc/gen_synth_syntax.py --out $D --n-train 350 --n-test 60 --k 8
echo "=== split: base = transforms 0-6, new = transform 7 (len_prefix) ==="
python qwen_poc/make_task_split.py --data $D --group-a 0,1,2,3,4,5,6 --out-a $BASE --out-b $NEW
echo "=== base backbone (Phase A on 7 base transforms, expert tokens frozen-random) ==="
$TRB --data $BASE --variant em --phase backbone --lr 1e-5 --steps 1000 --out $O/base_bb
echo "-- BASE_BB (before adding transform 7)"; NEWACC $O/base_bb; BASEACC $O/base_bb
echo "=== ARM 1: token-only (Phase B, backbone FROZEN — fit only the new expert-7 row) ==="
$TRB --data $NEW --variant em --phase tokens --lr 1e-2 --init-from $O/base_bb --steps 300 --save-steps 30,100 --out $O/tok
for s in 30 100; do echo "-- TOK s$s"; NEWACC $O/tok__s$s; done
echo "-- TOK s300"; NEWACC $O/tok; BASEACC $O/tok
echo "-- TOK s300 new-token SWAP (route expert-7 through wrong token)"; python qwen_poc/syntax_demo.py --run $O/tok --data $D --variant em --only 7 --swap --show 0 --max-eval 480 2>&1 | grep OVERALL
echo "=== ARM 2: full SFT (Phase full, backbone + token both train) ==="
$TRB --data $NEW --variant em --phase full --lr 1e-5 --init-from $O/base_bb --steps 300 --save-steps 30,100 --out $O/full
for s in 30 100; do echo "-- FULL s$s"; NEWACC $O/full__s$s; done
echo "-- FULL s300"; NEWACC $O/full; BASEACC $O/full
echo "=== ARM 3: EM two-phase (Phase A backbone 150 -> Phase B tokens 150) ==="
$TRB --data $NEW --variant em --phase backbone --lr 1e-5 --init-from $O/base_bb --steps 150 --save-steps 30,100 --out $O/emA
for s in 30 100; do echo "-- EMA s$s"; NEWACC $O/emA__s$s; done
echo "-- EMA s150"; NEWACC $O/emA; BASEACC $O/emA
$TRB --data $NEW --variant em --phase tokens --lr 1e-2 --init-from $O/emA --steps 150 --save-steps 30,100 --out $O/emB
for s in 30 100; do echo "-- EMB s$s"; NEWACC $O/emB__s$s; done
echo "-- EMB s150"; NEWACC $O/emB; BASEACC $O/emB
echo "=== data-efficiency: token-only, 100 steps, n new examples ==="
for n in 5 25 100; do
  Nn=$O/new_n$n; mkdir -p $Nn; cp $NEW/experts.json $NEW/control.test.jsonl $NEW/em.test.jsonl $Nn/
  shuf $NEW/em.train.jsonl | head -n $n > $Nn/em.train.jsonl
  shuf $NEW/control.train.jsonl | head -n $n > $Nn/control.train.jsonl
  python qwen_poc/train_sft.py --model $M --bs 8 --max-len 128 --data $Nn --variant em --phase tokens --lr 1e-2 --init-from $O/base_bb --steps 100 --out $O/tokn$n
  echo "-- TOKN n$n"; NEWACC $O/tokn$n
done
find $O -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
echo SYNINC_DONE
