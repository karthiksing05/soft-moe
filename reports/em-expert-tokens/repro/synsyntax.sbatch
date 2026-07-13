#!/bin/bash -l
#SBATCH -J synsyntax
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
D=/ptmp/karthiksing/soft-moe/synsyntax_data
O=/ptmp/karthiksing/soft-moe/conv/synsyntax; mkdir -p $O
echo "=== generate synthetic-syntax data ==="
python qwen_poc/gen_synth_syntax.py --out $D --n-train 350 --n-test 60 --k 8
TRB="python qwen_poc/train_sft.py --model $M --data $D --bs 8 --max-len 128"
echo "=== train CONTROL (generic marker, joint) ==="
$TRB --variant control --phase full --lr 1e-5 --steps 1500 --out $O/control
echo "=== train SFT (per-persona token, JOINT full FT) ==="
$TRB --variant em --phase full --lr 1e-5 --steps 1500 --out $O/sft
echo "=== train EM two-phase: Phase A backbone (tokens frozen) 1000 -> Phase B tokens (backbone frozen) 500 ==="
$TRB --variant em --phase backbone --lr 1e-5 --steps 1000 --out $O/emA
$TRB --variant em --phase tokens   --lr 1e-2 --init-from $O/emA --steps 500 --out $O/em
echo "###### PERPLEXITY + SWAP ######"
echo "----- CONTROL (generic token) -----"; python qwen_poc/eval_poc.py --run $O/control --data $D --variant control --base-model $M 2>&1 | grep -E "ppl|MACRO"
echo "----- SFT (per-persona token, joint) -----"; python qwen_poc/eval_poc.py --run $O/sft --data $D --variant em --base-model $M 2>&1 | grep -E "ppl|MACRO"
echo "----- EM (per-persona token, two-phase) -----"; python qwen_poc/eval_poc.py --run $O/em --data $D --variant em --base-model $M 2>&1 | grep -E "ppl|MACRO"
echo "###### EXACT-MATCH GENERATION ACCURACY ######"
echo "----- CONTROL -----";           python qwen_poc/syntax_demo.py --run $O/control --data $D --variant control
echo "----- SFT (right token) -----"; python qwen_poc/syntax_demo.py --run $O/sft --data $D --variant em
echo "----- SFT (WRONG token/swap) -----"; python qwen_poc/syntax_demo.py --run $O/sft --data $D --variant em --swap
echo "----- EM (right token) -----";  python qwen_poc/syntax_demo.py --run $O/em --data $D --variant em
echo "----- EM (WRONG token/swap) -----"; python qwen_poc/syntax_demo.py --run $O/em --data $D --variant em --swap
rm -rf $O/control $O/sft $O/emA $O/em
echo SYNSYNTAX_DONE
