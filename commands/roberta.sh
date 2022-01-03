
set -ex
export CUDA_VISIBLE_DEVICES=6,7
export OMP_NUM_THREADS=1
LR=1e-3
EPOCH=300000

NEG_WORD=" irrelevant"
POS_WORD=" relevant"

MAX_STEPS=1
Q=$1
NEG=1
LOG_STEP=100
EVAL_EVERY=100
BATCH_SIZE=2
model="roberta-large-soft-prompt"
ckpt="/data/private/huxiaomeng/pretrained_models/roberta-large"

python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=3119 \
train.py \
-task prompt_classification \
-model roberta \
-qrels /data/private/huxiaomeng/promptir/collections/msmarco-passage/qrels.train.tsv     \
-train /data/private/huxiaomeng/promptir/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
-dev /data/private/huxiaomeng/promptir/dataset/msmarco/dev/5-q.jsonl  \
-test /data/private/huxiaomeng/promptir/dataset/msmarco/test/all-q.jsonl  \
-max_input 80000000 \
-vocab $ckpt  \
-pretrain $ckpt  \
-metric mrr_cut_10  \
-max_query_len 50  \
-max_doc_len 400 \
-epoch $EPOCH  \
-batch_size $BATCH_SIZE  \
-lr $LR  \
-eval_every $EVAL_EVERY  \
-optimizer adamw   \
-dev_eval_batch_size 200  \
-n_warmup_steps 0  \
-logging_step $LOG_STEP  \
-save /data/private/huxiaomeng/promptir/checkpoints/$model/q$Q-n-$NEG/  \
-res /data/private/huxiaomeng/promptir/results/$model/q$Q-n-$NEG.trec  \
-test_res /data/private/huxiaomeng/promptir/results/$model/test_q$Q-n-$NEG.trec  \
--log_dir=/data/private/huxiaomeng/promptir/logs/$model/q$Q-n-$NEG/  \
--max_steps=$MAX_STEPS  \
--pos_word=" relevant"  \
--neg_word=" irrelevant"  \
--template='<q> <mask> <d>'  \
--soft_prompt  \
-gradient_accumulation_steps 1  \
--prefix='[133, 986, 3645, 16, 10, 25860, 4, 48360, 16, 1437]'   \
--infix='[11167,10]'    \
--suffix='[560, 22053, 257, 991, 3999, 4, 20, 220, 3645, 16, 10, 3780, 4]'   \
--soft_sentence=""  \

