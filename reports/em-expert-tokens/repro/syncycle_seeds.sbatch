#!/bin/bash -l
#SBATCH -J synseed
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
# Multi-seed confirmation of the headline: token-FIRST vs A-FIRST at EQUAL backbone budget (150 steps),
# with token-only (floor) and replay (ceiling). ONE fixed base backbone; vary only the incremental-add
# training seed -> isolates the ordering effect from base-backbone variance.
M=Qwen/Qwen2.5-3B
D=/ptmp/karthiksing/soft-moe/synseed_data
O=/ptmp/karthiksing/soft-moe/conv/synseed; mkdir -p $O
BASE=$O/base_data; NEW=$O/new_data; MIX=$O/mix_data
EVAL(){ n=$(python qwen_poc/syntax_demo.py --run "$1" --data $D --variant em --only 7             --show 0 --max-eval 480 2>&1 | grep OVERALL | grep -oE '[0-9.]+%');
        b=$(python qwen_poc/syntax_demo.py --run "$1" --data $D --variant em --only 0,1,2,3,4,5,6 --show 0 --max-eval 480 2>&1 | grep OVERALL | grep -oE '[0-9.]+%');
        echo "RESULT  $2  new=$n  base=$b"; rm -rf "$1"; }
echo "=== data + split + base+new mix ==="
python qwen_poc/gen_synth_syntax.py --out $D --n-train 350 --n-test 60 --k 8
python qwen_poc/make_task_split.py --data $D --group-a 0,1,2,3,4,5,6 --out-a $BASE --out-b $NEW
mkdir -p $MIX; cp $NEW/experts.json $NEW/*.test.jsonl $MIX/
cat $BASE/em.train.jsonl $NEW/em.train.jsonl | shuf > $MIX/em.train.jsonl
cat $BASE/control.train.jsonl $NEW/control.train.jsonl | shuf > $MIX/control.train.jsonl
echo "=== fixed base backbone (Phase A on 7 base transforms) ==="
python qwen_poc/train_sft.py --model $M --data $BASE --variant em --phase backbone --lr 1e-5 --steps 1000 --seed 0 --bs 8 --max-len 128 --out $O/base_bb
CY="python qwen_poc/train_cycle.py --model $M --bs 8 --max-len 128 --data $NEW --init-from $O/base_bb"
for s in 0 1 2; do
  echo "###### SEED $s ######"
  $CY --seed $s --schedule B300                    --out $O/tokonly_$s; EVAL $O/tokonly_$s "s$s:token-only(B300)"
  python qwen_poc/train_sft.py --model $M --data $NEW --variant em --phase full --lr 1e-5 --init-from $O/base_bb --steps 300 --seed $s --bs 8 --max-len 128 --out $O/full_$s; EVAL $O/full_$s "s$s:full-SFT(300)"
  $CY --seed $s --schedule A150,B150               --out $O/afirst_$s;  EVAL $O/afirst_$s  "s$s:A-first(A150,B150)"
  $CY --seed $s --schedule B100,A150,B40           --out $O/tfirst_$s;  EVAL $O/tfirst_$s  "s$s:token-first(B100,A150,B40)"
  $CY --seed $s --schedule A150,B150 --a-data $MIX --out $O/replay_$s;  EVAL $O/replay_$s  "s$s:REPLAY(A150 base+new,B150)"
done
rm -rf $O/base_bb $BASE $NEW $MIX
echo SYNSEED_DONE
