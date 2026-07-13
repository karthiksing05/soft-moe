#!/bin/bash -l
#SBATCH -J syncycle
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
# Cyclic EM sweep: can an A/B schedule add a NOVEL transform (len_prefix) to a 7-transform backbone with
# token-only-like base retention AND full-SFT-like new-transform accuracy? Mechanism: pre-fit token (B) so
# a backbone burst (A) sees low loss -> small gradients -> little drift -> little forgetting.
M=Qwen/Qwen2.5-3B
D=/ptmp/karthiksing/soft-moe/syncycle_data
O=/ptmp/karthiksing/soft-moe/conv/syncycle; mkdir -p $O
BASE=$O/base_data; NEW=$O/new_data; MIX=$O/mix_data
CY="python qwen_poc/train_cycle.py --model $M --bs 8 --max-len 128 --data $NEW --init-from $O/base_bb"
# RESULT <label> new=<x%> base=<y%>  — one parseable line per treatment
EVAL(){ n=$(python qwen_poc/syntax_demo.py --run "$1" --data $D --variant em --only 7             --show 0 --max-eval 480 2>&1 | grep OVERALL | grep -oE '[0-9.]+%');
        b=$(python qwen_poc/syntax_demo.py --run "$1" --data $D --variant em --only 0,1,2,3,4,5,6 --show 0 --max-eval 480 2>&1 | grep OVERALL | grep -oE '[0-9.]+%');
        echo "RESULT  $2  new=$n  base=$b"; rm -rf "$1"; }
echo "=== data (K=8) + split base 0-6 / new 7 (len_prefix) + base+new mix for replay ==="
python qwen_poc/gen_synth_syntax.py --out $D --n-train 350 --n-test 60 --k 8
python qwen_poc/make_task_split.py --data $D --group-a 0,1,2,3,4,5,6 --out-a $BASE --out-b $NEW
mkdir -p $MIX; cp $NEW/experts.json $NEW/*.test.jsonl $MIX/
cat $BASE/em.train.jsonl $NEW/em.train.jsonl | shuf > $MIX/em.train.jsonl
cat $BASE/control.train.jsonl $NEW/control.train.jsonl | shuf > $MIX/control.train.jsonl
echo "=== base backbone: Phase A on 7 base transforms (expert tokens frozen-random) ==="
python qwen_poc/train_sft.py --model $M --data $BASE --variant em --phase backbone --lr 1e-5 --steps 1000 --bs 8 --max-len 128 --out $O/base_bb
python qwen_poc/syntax_demo.py --run $O/base_bb --data $D --variant em --only 7 --show 0 --max-eval 480 2>&1 | grep OVERALL | sed 's/^/RESULT  base_bb(before) new=/'
python qwen_poc/syntax_demo.py --run $O/base_bb --data $D --variant em --only 0,1,2,3,4,5,6 --show 0 --max-eval 480 2>&1 | grep OVERALL | sed 's/^/        base_bb(before) base=/'
echo "###### REFERENCES ######"
$CY --schedule B300                    --out $O/r_tokonly;  EVAL $O/r_tokonly  "ref:token-only(B300)"
python qwen_poc/train_sft.py --model $M --data $NEW --variant em --phase full --lr 1e-5 --init-from $O/base_bb --steps 300 --bs 8 --max-len 128 --out $O/r_full; EVAL $O/r_full "ref:full-SFT(300)"
$CY --schedule A150,B150               --out $O/r_em1;      EVAL $O/r_em1      "ref:EM-1cycle(A150-first,B150)"
$CY --schedule A150,B150 --a-data $MIX --out $O/r_replay;   EVAL $O/r_replay   "ref:REPLAY(A150 base+new,B150)"
echo "###### Q1: token-first Pareto frontier vs backbone budget a  (B100, A{a}, B40) ######"
for a in 0 10 20 40 80 150; do
  $CY --schedule B100,A${a},B40 --out $O/q1_a$a; EVAL $O/q1_a$a "q1:tokfirst-a${a}(B100,A${a},B40)"
done
echo "###### Q2: granularity at fixed total backbone budget A=80, token-first, refit between ######"
$CY --schedule B100,A80,B40                                              --out $O/q2_n1; EVAL $O/q2_n1 "q2:N1(A80x1)"
$CY --schedule B100,A40,B20,A40,B20                                      --out $O/q2_n2; EVAL $O/q2_n2 "q2:N2(A40x2)"
$CY --schedule B100,A20,B15,A20,B15,A20,B15,A20,B15                      --out $O/q2_n4; EVAL $O/q2_n4 "q2:N4(A20x4)"
$CY --schedule B100,A10,B10,A10,B10,A10,B10,A10,B10,A10,B10,A10,B10,A10,B10,A10,B10 --out $O/q2_n8; EVAL $O/q2_n8 "q2:N8(A10x8)"
rm -rf $O/base_bb $BASE $NEW $MIX
echo SYNCYCLE_DONE
