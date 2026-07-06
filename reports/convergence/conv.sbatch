#!/bin/bash -l
#SBATCH -o /ptmp/karthiksing/soft-moe/logs/%x_%j.out
#SBATCH -e /ptmp/karthiksing/soft-moe/logs/%x_%j.err
#SBATCH -D /u/karthiksing/projects/soft-moe
#SBATCH -A mpib_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=05:00:00
module purge; module load python-waterboa/2025.06
source /ptmp/karthiksing/soft-moe-venv/bin/activate
export HF_HOME=/ptmp/karthiksing/hf_cache PYTHONPATH=.
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
set -e
M=$1; D=$2; EVAL=$3; MAXLEN=$4; LR=$5; BS=$6; O=$7
mkdir -p $O
TR="python qwen_poc/train_sft.py --model $M --data $D --bs $BS --max-len $MAXLEN"
EV(){ python qwen_poc/$EVAL --run "$1" --data $D --variant "$2" --base-model $M 2>&1 | grep -E "MACRO|acc|ppl" | tail -3; }
echo "###### CONVERGENCE  M=$M  D=$D ######"
echo "=== ARM sft_gen (generic-token standard SFT), 3000 steps ==="
$TR --variant control --phase full --lr $LR --steps 3000 --save-steps 250,500,1000,2000 --out $O/sft_gen
for s in 250 500 1000 2000; do echo "-- sft_gen s$s"; EV $O/sft_gen__s$s control; done; echo "-- sft_gen s3000"; EV $O/sft_gen control
echo "=== ARM sft_tok (expert-token joint SFT), 3000 steps ==="
$TR --variant em --phase full --lr $LR --steps 3000 --save-steps 250,500,1000,2000 --out $O/sft_tok
for s in 250 500 1000 2000; do echo "-- sft_tok s$s"; EV $O/sft_tok__s$s em; done; echo "-- sft_tok s3000"; EV $O/sft_tok em
echo "=== ARM emA (EM Phase A: backbone, frozen distinct tokens), 2000 steps ==="
$TR --variant em --phase backbone --lr $LR --steps 2000 --save-steps 500,1000,1500,1750 --out $O/emA
for s in 500 1000 1500 1750; do echo "-- emA s$s (B=0)"; EV $O/emA__s$s em; done; echo "-- emA s2000 (B=0)"; EV $O/emA em
echo "=== EM split sweep at total T=2000 (Phase B tokens-only from each Phase-A ckpt) ==="
for pair in "1750 250" "1500 500" "1000 1000" "500 1500"; do
  set -- $pair; A=$1; B=$2; SRC=$O/emA__s$A
  echo "-- split A=$A B=$B"
  $TR --variant em --phase tokens --lr 1e-2 --init-from $SRC --steps $B --out $O/emB_A${A}_B${B}
  EV $O/emB_A${A}_B${B} em
done
echo "=== cleanup model dirs, keep *.json ==="
find $O -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
echo "CONV_DONE $O"
