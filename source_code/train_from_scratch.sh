TIME=`date +"%Y_%m_%d_%H_%M_%S"`
TASK=bart

TOTAL_NUM_UPDATES=42000
WARMUP_UPDATES=10
LR=1e-4
MAX_TOKENS=2000
BATCH_SIZE=80
UPDATE_FREQ=1
EPOCHS=15

tensorboard_path=tensorboard/$TASK/$TIME

#BART_PATH=/path/to/bart/model.pt


CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/$TASK \
    --tensorboard-logdir $tensorboard_path \
    --max-epoch $EPOCHS \
    --batch-size $BATCH_SIZE \
    --save-interval 1 \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_base \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --z-size 256 \
    --init-w 0.02;
#    --restore-file $BART_PATH \


