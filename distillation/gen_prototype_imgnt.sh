python gen_prototype.py \
    --batch_size 10 \
    --data_dir /data2/wlf/datasets/imagenet \
    --dataset imagenet \
    --diffusion_checkpoints_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --ipc 10 \
    --km_expand 1 \
    --label_file_path ./label-prompt/imagenet_classes.txt \
    --save_prototype_path ./prototypes
