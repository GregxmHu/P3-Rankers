
set -ex
export CUDA_VISIBLE_DEVICES=6,7,4,2
export OMP_NUM_THREADS=1
LR=2e-5
EPOCH=30000


MAX_STEPS=3000
Q=5
NEG=1
LOG_STEP=100
EVAL_EVERY=300
BATCH_SIZE=8
model="bert-large-auto-prompt"
ckpt="/data/private/huxiaomeng/pretrained_models/bert-large"
dir_path="/data/private/huxiaomeng/promptir"
python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=2119 \
train.py \
-task prompt_classification \
-model bert \
-qrels $dir_path/collections/msmarco-passage/qrels.train.tsv     \
-train $dir_path/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
-dev $dir_path/dataset/msmarco/dev/5-q.jsonl  \
-test $dir_path/dataset/msmarco/test/all-q.jsonl  \
-max_input 80000000 \
-vocab $ckpt  \
-pretrain $ckpt  \
-metric mrr_cut_100  \
-max_query_len 50  \
-max_doc_len 400 \
-epoch $EPOCH  \
-batch_size $BATCH_SIZE  \
-lr $LR  \
-eval_every $EVAL_EVERY  \
-optimizer adamw   \
-dev_eval_batch_size 128  \
-n_warmup_steps 0  \
-logging_step $LOG_STEP  \
-save $dir_path/checkpoints/$model/q$Q-n-$NEG/  \
-res $dir_path/results/$model/q$Q-n-$NEG.trec  \
-test_res $dir_path/results/$model/test_q$Q-n-$NEG.trec  \
--log_dir=$dir_path/logs/$model/q$Q-n-$NEG/  \
--max_steps=$MAX_STEPS  \
--pos_word="relevant"  \
--neg_word="irrelevant"  \
--template='<q>? Is this [MASK] to your situation? <d>'  \
-gradient_accumulation_steps 1  \
--prefix='[133, 986, 3645, 16, 10, 25860, 4, 48360, 16, 1437]'   \
--infix='[11167,10]'    \
--suffix='[560, 22053, 257, 991, 3999, 4, 20, 220, 3645, 16, 10, 3780, 4]'   \
--soft_sentence=""  \






