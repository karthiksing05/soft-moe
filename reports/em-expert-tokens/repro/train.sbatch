#!/bin/bash -l
#SBATCH -J poc_train
#SBATCH -o /ptmp/karthiksing/soft-moe/logs/%x_%j.out
#SBATCH -D /u/karthiksing/projects/soft-moe
#SBATCH -A mpib_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=02:30:00
module purge; module load python-waterboa/2025.06
source /ptmp/karthiksing/soft-moe-venv/bin/activate
export HF_HOME=/ptmp/karthiksing/hf_cache; export PYTHONPATH=.
python reports/qwen_poc/train_sft.py "$@"
