set -ex
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
LR=1

MAX_STEPS=3000
EPOCH=300000

Q=50
LOG_STEP=50
EVAL_EVERY=100
 
BATCH_SIZE=2
NEG=1

#model="t5-lm-adapt-soft-prompt"
model="t5-v11-soft-prompt"
#ckpt="/mnt/101_data1/private/hxm/pretrained_models/t5-lm-adapt-large"
ckpt="/mnt/101_data1/private/hxm/pretrained_models/t5-v11-large"
python -m torch.distributed.launch \
         --nproc_per_node=4 \
         --master_port=10627  \
        train.py \
        -task classification  \
        -model t5  \
        -qrels /mnt/101_data1/private/hxm/promptir/collections/msmarco-passage/qrels.train.tsv     \
        -train /mnt/101_data1/private/hxm/promptir/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
        -dev /mnt/101_data1/private/hxm/promptir/dataset/msmarco/dev/50-q.jsonl  \
        -test /mnt/101_data1/private/hxm/promptir/dataset/msmarco/test/all-q.jsonl  \
        -max_input 80000000  \
        -vocab $ckpt          \
        -pretrain $ckpt  \
        -save /mnt/101_data1/private/hxm/promptir/checkpoints/$model/q$Q-n-$NEG/  \
        -res /mnt/101_data1/private/hxm/promptir/results/$model/q$Q-n-$NEG.trec  \
        -test_res /mnt/101_data1/private/hxm/promptir/results/$model/test_q$Q-n-$NEG.trec  \
        --log_dir=/mnt/101_data1/private/hxm/promptir/logs/$model/q$Q-n-$NEG/  \
        -metric mrr_cut_10  \
        -max_query_len 76  \
        -max_doc_len 290  \
        -epoch $EPOCH  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -eval_every $EVAL_EVERY  \
        -optimizer adamw  \
        -dev_eval_batch_size  64   \
        -n_warmup_steps 0  \
        -logging_step $LOG_STEP  \
        --max_steps=$MAX_STEPS \
        -gradient_accumulation_steps 4 \
        --soft_sentence=""  \
        --template="Task: Find the relevance between Query and Document. Query: <q> Document: <d> Relevant: "   \
        --prefix='[16107, 10, 2588, 8, 20208, 344, 3, 27569, 11, 11167, 5,3,27569,10]'   \
        --infix='[11167,10]'    \
        --suffix='[31484,17,10,1]'   \
        --soft_prompt   \
        
                                                                                                                                                                                                          

