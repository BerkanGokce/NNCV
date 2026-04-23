wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.0012 \
    --weight-decay 1e-4 \
    --momentum 0.9 \
    --num-workers 10 \
    --seed 42 \
    --resize-h 512 \
    --resize-w 1024 \
    --scale-min 0.75 \
    --scale-max 1.5 \
    --hflip-prob 0.5 \
    --aux-weight 0.3 \
    --experiment-id "bisenetv2-efficiency-v2"