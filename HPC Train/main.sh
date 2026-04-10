wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.005 \
    --weight-decay 1e-4 \
    --momentum 0.9 \
    --num-workers 10 \
    --seed 42 \
    --resize-h 512 \
    --resize-w 1024 \
    --scale-min 0.75 \
    --scale-max 1.75 \
    --hflip-prob 0.5 \
    --experiment-id "deeplabv3-resnet50-peak" \
