python train_FKD.py \
    --batch-size 1024 \
    --model resnet18 \
    --cos \
    -j 8 --gradient-accumulation-steps 2 \
    -T 20 \
    --mix-type 'cutmix' \
    --wandb-api-key ceb6f265b99ab43eba47ae94afbbd639d54b3153 \
    --output-dir ./save/final_rn18_fkd/tiny_imagenet_ipc50_label18_train182 \
    --train-dir ../data/distilled_data/tiny_imagenet_ipc50_50_s0.7_g8.0_kmexpand1 \
    --val-dir /data2/wlf/datasets/tiny-imagenet/val \
    --fkd-path ../matching/tiny_imagenet_ipc50_label182 \
    --wandb-project tiny_imagenet_ipc50_label182

# tiny-imagenet

python train_FKD.py     --batch-size 128     --model resnet18     --cos     -j 8 --gradient-accumulation-steps 2     -T 20     --mix-type 'cutmix'     --wandb-api-key ceb6f265b99ab43eba47ae94afbbd639d54b3153     --output-dir ./save/final_rn18_fkd/tiny_imagenet_ipc50_label18_train182     --train-dir ../data/distilled_data/tiny_imagenet_ipc50_50_s0.7_g8.0_kmexpand1     --val-dir /data2/wlf/datasets/tiny-imagenet/val     --fkd-path ../matching/tiny_imagenet_ipc50_label182     --wandb-project tiny_imagenet_ipc50_label182
