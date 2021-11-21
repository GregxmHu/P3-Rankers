
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
LR=1
EPOCH=300000

NEG_WORD=" irrelevant"
POS_WORD=" relevant"

MAX_STEPS=5000
Q=1000
NEG=1
LOG_STEP=100
EVAL_EVERY=100
BATCH_SIZE=4
model="test_roberta"
ckpt="/mnt/101_data1/private/hxm/pretrained_models/roberta-large"

python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=3119 \
train.py \
-task prompt_classification \
-model roberta \
-qrels /mnt/101_data1/private/hxm/promptir/collections/msmarco-passage/qrels.train.tsv     \
-train /mnt/101_data1/private/hxm/promptir/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
-dev /mnt/101_data1/private/hxm/promptir/dataset/msmarco/dev/500-q.jsonl  \
-test /mnt/101_data1/private/hxm/promptir/dataset/msmarco/test/all-q.jsonl  \
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
-dev_eval_batch_size 42  \
-n_warmup_steps 0  \
-logging_step $LOG_STEP  \
-save /mnt/101_data1/private/hxm/promptir/checkpoints/$model/q$Q-n-$NEG/  \
-res /mnt/101_data1/private/hxm/promptir/results/$model/q$Q-n-$NEG.trec  \
-test_res /mnt/101_data1/private/hxm/promptir/results/$model/test_q$Q-n-$NEG.trec  \
--log_dir=/mnt/101_data1/private/hxm/promptir/logs/$model/q$Q-n-$NEG/  \
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

