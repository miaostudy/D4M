python gen_syn_image.py \
    --dataset tiny_imagenet \
    --diffusion_checkpoints_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --guidance_scale 8 \
    --strength 0.7 \
    --ipc 50 \
    --km_expand 1 \
    --label_file_path ./label-prompt/tinyimgnt-label.txt \
    --prototype_path ./prototypes/tiny_imagenet-ipc50-kmexpand1.json \
    --save_init_image_path ../data/distilled_data/
