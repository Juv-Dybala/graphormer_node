# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0 fairseq-train \
--user-dir ../../graphormer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name BlogCatalog \
--dataset-source pyg \
--task node_prediction \
--criterion node_multiclass_cross_entropy \
--arch graphormer_base \
--num-classes 6 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 16 \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 768 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 32 \
--max-epoch 200 \
--save-dir ./ckpts/Blog \
--seed 42 \
--save-interval 20 \
# --keep-last-checkpoints