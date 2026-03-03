import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Sequence, List, Tuple
from dataclasses import dataclass
import random
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_train_val_datasets(tokenizer, test_size=0.2, random_state=42):

    reg_train_df = pd.read_csv("/home/wangaw/ESM/semi_supervised/code_submission/round2_semi_sup/data/reg_train_and_generate_data.csv")
    reg_val_df = pd.read_csv("data/reg_val.csv")

    train_datasets = {
        'reg': AMPRegDataset(
            df=reg_train_df,
            tokenizer=tokenizer
        )
    }
    
    val_datasets = {
        'reg': AMPRegDataset(
            df=reg_val_df,
            tokenizer=tokenizer
        ),
        
    }
    
    # 打印数据集大小信息
    print("\n数据集拆分信息：")
    for name in ['reg']:
        print(f"\n{name.upper()} 数据集:")
        print(f"训练集大小: {len(train_datasets[name])}")
        print(f"验证集大小: {len(val_datasets[name])}")
    
    return train_datasets, val_datasets

class AMPDataset(Dataset):
    def __init__(self, df,  tokenizer, max_length=40):
        """
        Args:
            csv_path (str): CSV文件路径。
            structure_features_dir (str): 结构化特征文件的目录。
            tokenizer: 预训练tokenizer。
            max_length (int): 序列的最大长度，用于填充和截断。
        """
        
        self.df = df.copy()
        if "Sequence" not in self.df.columns:
            raise ValueError("CSV中缺少列 'Sequence'")
        self.df = self.df[self.df["Sequence"].notna()].reset_index(drop=True)
        self.df["Sequence"] = self.df["Sequence"].astype(str).str.strip()
        self.df = self.df[self.df["Sequence"] != ""].reset_index(drop=True)

        self.sequences = self.df["Sequence"].tolist()
        self.labels = self._extract_labels(self.df)
        self.max_length = max_length
        self.tokenizer = tokenizer

    def _extract_labels(self, df):
        """
        从CSV文件中提取标签。
        如果 'label' 列存在，则返回其列表；否则返回 None。

        Args:
            df (pd.DataFrame): 读取的CSV文件。

        Returns:
            list or None: 标签列表或None。
        """
        # 检查 'label' 是否是DataFrame的列名之一
        if "label" in df.columns:
            return df["label"].tolist()
        else:
            # 如果 'label' 列不存在，则返回 None
            return None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # Tokenize序列
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        input_ids = encoding["input_ids"].squeeze(0)  # 去掉batch维度
        attention_mask = encoding["attention_mask"].squeeze(0)  # 去掉batch维度
        # 加载结构化特征
        # 返回数据
        return input_ids, attention_mask, label


class AMPRegDataset(AMPDataset):
    """抗菌肽MIC回归任务数据集"""
    def _extract_labels(self, df):
        return df["label"].tolist()

    def __getitem__(self, idx):
        input_ids, attention_mask,label = super().__getitem__(idx)
        return input_ids, attention_mask,torch.tensor(label, dtype=torch.float)


from torch.nn.utils.rnn import pad_sequence
import torch

class AMPCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # batch: list of tuples -> (input_ids, attention_mask, label_float_tensor)
        input_ids = [item[0].long() for item in batch]
        attention_mask = [item[1].long() for item in batch]
        labels = [item[2].float() for item in batch]  # 每条是 0-dim float tensor

        # pad 到相同长度
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # 拼成一维 [B] 标签张量
        labels = torch.stack(labels).float()  # shape: [batch]

        return {
            "input_ids": input_ids,           # [B, L]
            "attention_mask": attention_mask, # [B, L]
            "labels": labels                  # [B]
        }