#!/bin/bash -l
#SBATCH -J scaleinc
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
export HF_HOME=/ptmp/karthiksing/hf_cache PYTHONPATH=. HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e
M=Qwen/Qwen2.5-3B
POOL=/ptmp/karthiksing/soft-moe/persona_pool64
O=/ptmp/karthiksing/soft-moe/conv/scaleinc; mkdir -p $O
NEW3=$O/new3
python qwen_poc/filter_pool.py --data $POOL --out $NEW3 --ids 61,62,63
TRB="python qwen_poc/train_sft.py --model $M --bs 8 --max-len 512"
EV(){ python qwen_poc/eval_poc.py --run "$1" --data "$2" --variant em --base-model $M 2>&1 | grep -E "MACRO|ppl"; }
echo "###### INCREMENTAL SCALING: new-persona ppl (token-only) vs #base personas K ######"
for K in 2 4 8 16 32 60; do
  IDS=$(python -c "print(','.join(str(i) for i in range($K)))")
  BK=$O/base_$K
  python qwen_poc/filter_pool.py --data $POOL --out $BK --ids $IDS
  echo "===== K=$K base personas ====="
  $TRB --data $BK --variant em --phase backbone --lr 1e-5 --steps 1000 --out $O/bb_$K
  $TRB --data $NEW3 --variant em --phase tokens --lr 1e-2 --init-from $O/bb_$K --steps 100 --out $O/inc_$K
  echo "-- INC K=$K (add new personas 61,62,63 via token-only)"; EV $O/inc_$K $NEW3
  rm -rf $O/bb_$K $O/inc_$K $BK
done
echo SCALEINC_DONE
