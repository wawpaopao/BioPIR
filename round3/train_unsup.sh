
export CUDA_VISIBLE_DEVICES=7
lr=1e-5
seed=42  # 将种子值设置为固定的值

fold_num=1

output_dir="/data/wangaw/ESM/round3_contrastive/fold_${fold_num}"
mkdir -p "$output_dir"

python unsup_fold1.py  \
        --run_name "Hierarchical_${lr}_rna_data0_seed${seed}_fold${fold_num}" \
        --model_max_length 40 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 \
        --learning_rate "${lr}" \
        --num_train_epochs 10 \
        --seed "${seed}" \
        --fp16 \
        --weight_decay 0.01 \
        --save_steps 2000 \
        --output_dir "$output_dir" \
        --lr_scheduler_type cosine \
        --warmup_steps 4000 \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --eval_steps 2000 \
        --evaluation_strategy steps

 


