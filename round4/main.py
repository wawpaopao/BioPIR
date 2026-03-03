
import yaml
import torch
from model import MultiTaskPeptideModel, AdaptiveContrastiveLoss
from dataset import ContrastivePeptideDataset,MICPeptideDataset , DataCollator,load_and_split_peptide_data
from trainer import ContrastiveTrainer
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
import pandas as pd
import os
import random
import numpy as np
import wandb
import datetime
import torch.nn as nn
def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config():
    with open("config_contrastive_only.yaml") as f:
        return yaml.safe_load(f)['training']


def main():
    config = load_config()
    set_seed(config.get('seed', 42))  # 设置随机种子
    
    # 加载ESM模型和分词器
    MODEL_NAME_OR_PATH = '/data/wangaw/ESM/esm_model'
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        padding_side='right',
        use_fast=True,
        model_max_length=config.get('max_seq_length', 40),
        trust_remote_code=True,
    )
    
    esm_model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

    # 加载数据
    
    (
        train_contrastive_sequences, train_contrastive_labels,
        val_contrastive_sequences, val_contrastive_labels,
        train_mic_sequences, train_mic_labels,
        val_mic_sequences, val_mic_labels
    ) = load_and_split_peptide_data(
        config['contrastive_data_path'],
        config['mic_data_path'],
        num_pair=60000,
        num_mic=10000,
        val_ratio=config['val_ratio']
    )
   
    print("训练集对比学习数据：", len(train_contrastive_sequences))
    print("验证集对比学习数据：", len(val_contrastive_sequences))
    print("训练集 MIC 数据：", len(train_mic_sequences))
    print("验证集 MIC 数据：", len(val_mic_sequences))
    
    
    # 创建数据集
    collator = DataCollator()
    
    # 创建模型
    model = MultiTaskPeptideModel(
        esm_model=esm_model,
        projection_dim=config.get('projection_dim', 256),
        dropout=config.get('dropout', 0.1),
        structure_feature_dim=9,
        freeze_base_model=config.get('freeze_base_model', False)
    )
    
    # 创建损失函数
    contrastive_loss_fn = AdaptiveContrastiveLoss(
        initial_margin=config.get('initial_margin', 1.0),
        hard_weight=config.get('hard_weight', 2.0),
        margin_update_factor=config.get('margin_update_factor', 0.05)
    )
    
    # 设置模型的损失函数
    mic_loss_fn = nn.MSELoss()
    
    # 初始化wandb
    use_wandb = config.get('use_wandb', False)
    if use_wandb:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        default_run_name = f"contrastive-training-{current_time}"
        
        wandb.init(
            project=config.get('wandb_project', 'peptide_multitask_mic_contrastive_learning'),
            name=config.get('wandb_run_name', default_run_name),
            config=config
        )
    
        # 第一阶段：基础对比学习
    print("=== 准备第一阶段训练数据 ===")
    train_contrastive_dataset = ContrastivePeptideDataset(
            sequence_pairs=train_contrastive_sequences,
            labels=train_contrastive_labels,
            tokenizer=tokenizer,
            structure_features_dir = config.get('structure_data_path'),
            max_length=config.get('max_seq_length', 40)
        )
        
    val_contrastive_dataset = ContrastivePeptideDataset(
            sequence_pairs=val_contrastive_sequences,
            labels=val_contrastive_labels,
            tokenizer=tokenizer,
            structure_features_dir = config.get('structure_data_path'),
            max_length=config.get('max_seq_length', 40)
        )
    
    train_mic_dataset = MICPeptideDataset(
            sequences=train_mic_sequences,
            mic_values=train_mic_labels,
            tokenizer=tokenizer,
            structure_features_dir = config.get('structure_data_path'),
            max_length=config.get('max_seq_length', 40)
    )

    val_mic_dataset = MICPeptideDataset(
            sequences=val_mic_sequences,
            mic_values=val_mic_labels,
            tokenizer=tokenizer,
            structure_features_dir = config.get('structure_data_path'),
            max_length=config.get('max_seq_length', 40)
    )

    trainer = ContrastiveTrainer(
        model=model, 
        mic_train_dataset = train_mic_dataset,
        contrastive_train_dataset = train_contrastive_dataset,
        contrastive_loss_fn = contrastive_loss_fn,
        mic_loss_fn = mic_loss_fn, 
        mic_val_dataset = val_mic_dataset,
        contrastive_val_dataset=val_contrastive_dataset,
        config=config
    )

        
    results = trainer.train(
        epochs=config.get('epochs'),
        contrastive_weight=config.get('contrastive_weight')
    )
    # 关闭wandb
    if use_wandb:
        wandb.finish()
    
    print("训练完成！")
    return results

if __name__ == "__main__":
    main()
