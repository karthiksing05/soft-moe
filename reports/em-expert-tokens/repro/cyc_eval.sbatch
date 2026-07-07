#!/bin/bash -l
#SBATCH -J cyc_eval
#SBATCH -o /ptmp/karthiksing/soft-moe/logs/%x_%j.out
#SBATCH -D /u/karthiksing/projects/soft-moe
#SBATCH -A mpib_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=00:35:00
module purge; module load python-waterboa/2025.06
source /ptmp/karthiksing/soft-moe-venv/bin/activate
export HF_HOME=/ptmp/karthiksing/hf_cache; export PYTHONPATH=.
K=$1; O=/ptmp/karthiksing/soft-moe/cyc_runs/K$K
declare -A DD=( [5]=K5 [15]=K16 [28]=K28 ); D=/ptmp/karthiksing/soft-moe/sweep_data/${DD[$K]}
echo "==== K=$K CONTROL ===="; python reports/qwen_poc/eval_poc.py --run $O/control --data $D --variant control 2>&1 | grep MACRO
echo "==== K=$K EM (2 cycles) ===="; python reports/qwen_poc/eval_poc.py --run $O/em_B --data $D --variant em 2>&1 | grep -E "MACRO|swap"
echo "CYC_EVAL_K${K}_DONE"
