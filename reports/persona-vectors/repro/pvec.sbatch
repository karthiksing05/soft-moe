#!/bin/bash -l
#SBATCH -J pvec
#SBATCH -o /ptmp/karthiksing/soft-moe/logs/%x_%j.out
#SBATCH -e /ptmp/karthiksing/soft-moe/logs/%x_%j.err
#SBATCH -D /u/karthiksing/projects/soft-moe
#SBATCH -A mpib_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=04:00:00
module purge; module load python-waterboa/2025.06
source /ptmp/karthiksing/soft-moe-venv/bin/activate
export HF_HOME=/ptmp/karthiksing/hf_cache PYTHONPATH=. HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e
M=Qwen/Qwen2.5-3B
PD=/ptmp/karthiksing/soft-moe/persona_data_big; PO=/ptmp/karthiksing/soft-moe/conv/pvec/persona
DD=/ptmp/karthiksing/soft-moe/domain_data;       DO=/ptmp/karthiksing/soft-moe/conv/pvec/domain
O=/ptmp/karthiksing/soft-moe/conv/pvec; mkdir -p $O
echo "=== PERSONA: train EM + analyse ==="
python qwen_poc/train_sft.py --model $M --data $PD --bs 8 --max-len 512 --variant em --phase backbone --lr 1e-5 --steps 1000 --out $PO/bb
python qwen_poc/train_sft.py --model $M --data $PD --bs 8 --max-len 512 --variant em --phase tokens   --lr 1e-2 --init-from $PO/bb --steps 500 --out $PO/em
python qwen_poc/analyze_token_space.py --run $PO/em --data $PD --out $O/persona_tokspace.json
python qwen_poc/compose_personas.py    --run $PO/em --data $PD --pairs pirate-robot,coach-child,bard-detective --out $O/persona_compose.json
rm -rf $PO
echo "=== DOMAIN: train EM (bs 4, max-len 1024) + analyse ==="
python qwen_poc/train_sft.py --model $M --data $DD --bs 4 --max-len 1024 --variant em --phase backbone --lr 1e-5 --steps 1000 --out $DO/bb
python qwen_poc/train_sft.py --model $M --data $DD --bs 4 --max-len 1024 --variant em --phase tokens   --lr 1e-2 --init-from $DO/bb --steps 500 --out $DO/em
python qwen_poc/analyze_token_space.py --run $DO/em --data $DD --out $O/domain_tokspace.json
python qwen_poc/compose_personas.py    --run $DO/em --data $DD --pairs math-medical,trivia-science --max-len 1024 --out $O/domain_compose.json
rm -rf $DO
echo PVEC_DONE
