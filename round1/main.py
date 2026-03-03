
import yaml
import torch
from model import RegressionModel
from dataset import create_train_val_datasets,AMPCollator
from trainer import RegressionTrainer
from transformers import AutoTokenizer, AutoModel,AutoModelForMaskedLM
from accelerate import Accelerator
import pandas as pd
import os
import random
import numpy as np
import wandb
import datetime
import torch.nn as nn
def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)['training']

def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))

def report_params(model, name="model"):
    try:
        from accelerate import Accelerator
        
    except Exception:
        pass
    trainable = count_params(model, trainable_only=True)
    total = count_params(model, trainable_only=False)
    ratio = (trainable / total) if total > 0 else 0.0
    print(f"{name} params: trainable={trainable/1e6:.2f}M ({trainable:,}), "
          f"total={total/1e6:.2f}M ({total:,}), trainable_ratio={ratio:.2%}")
    return trainable, total

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

    train_datasets, val_datasets = create_train_val_datasets(
        tokenizer=tokenizer,
        test_size=0.1,  # 可以调整验证集比例
        random_state=42
    )
    
    collator = AMPCollator(pad_token_id=tokenizer.pad_token_id)
    # 创建模型
    model = RegressionModel(
        esm_model=esm_model,
        projection_dim=config.get('projection_dim', 256),
        dropout=config.get('dropout', 0.1),
        freeze_base_model=False
    )

    report_params(model, name="RegressionModel")

    trainer = RegressionTrainer(model, train_datasets, val_datasets, collator,config)

    trainer.train(num_epochs=config.get('num_epochs', 10))

    
    print("训练完成！")
    

if __name__ == "__main__":
    main()
