export CUDA_VISIBLE_DEVICES=2

accelerate launch --config_file accelerate_singleNode_config.yaml main.py
