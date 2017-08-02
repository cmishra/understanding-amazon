jbsub -interactive -queue x86_6h -cores 8+0 -mem 5G \
    python runs/train.py \
    --data data_processed \
    --cache_filepath epoch_cache \
    --model alexnet \
    --workers 1 \
    --epochs 1 \
    --batch_size 64 \
    --optimizer adam \
    --momentum 0.0 \
    --lr_schedule Constant \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --criterion mse
