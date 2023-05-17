CUDA_VISIBLE_DEVICES=0 python main.py --data_corruption ./ImageNet-C \
--exp_type label_shifts --method vct --model vitbase_timm --output ./output/label_shifts \
--cls_token_lr 0.005 --instance_token_lr 0.01 --seed 2023 --level 5