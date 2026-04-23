wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --momentum 0.9 \
    --num-workers 10 \
    --seed 42 \
    --resize-h 512 \
    --resize-w 1024 \
    --scale-min 0.8 \
    --scale-max 1.4 \
    --hflip-prob 0.5 \
    --aux-weight 0.2 \
    --experiment-id "bisenetv2-efficiency-v3"