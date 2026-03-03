import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Sequence, List, Tuple
from dataclasses import dataclass
import random
import pandas as pd

def load_and_split_peptide_data(contrastive_data_path, mic_data_path, num_pair=200000, num_mic=None, val_ratio=0.1):
    """
    加载肽序列数据（对比学习数据和 MIC 数据），并划分为训练集和验证集
    
    参数:
        contrastive_data_path: 对比学习数据的路径
        mic_data_path: MIC 数据的路径
        num_pair: 对比学习数据的数量（正负样本对的总数）
        num_mic: MIC 数据的数量
        val_ratio: 验证集的比例
    
    返回:
        train_contrastive_data: 训练集的对比学习数据 (序列对和标签)
        val_contrastive_data: 验证集的对比学习数据 (序列对和标签)
        train_mic_data: 训练集的 MIC 数据 (序列和标签)
        val_mic_data: 验证集的 MIC 数据 (序列和标签)
    """
    # 加载对比学习数据
    contrastive_data = []
    
    with open(contrastive_data_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                seq1, seq2, label = parts
                label = int(label)
                contrastive_data.append((seq1, seq2, label))
    
    # 统计正负样本数量
    pos_count = sum(1 for _, _, label in contrastive_data if label == 1)
    neg_count = sum(1 for _, _, label in contrastive_data if label == 0)
    print(f"加载了 {pos_count} 个正样本对和 {neg_count} 个负样本对")
    
    # 加载 MIC 数据，csv
    mic_data = pd.read_csv(mic_data_path)
    mic_data_list = list(zip(mic_data['Sequence'].tolist(), mic_data['label'].tolist()))
    
    print(f"加载了 {len(mic_data_list)} 个 MIC 数据样本")
    

    if num_pair is not None:
        if num_pair > 50000:
            # 取前50000个
            first_part = contrastive_data[:50000]
            remaining_data = contrastive_data[50000:]
            
            # 从剩余数据中随机选取 (num_pair - 50000) 个
            if len(remaining_data) > (num_pair - 50000):
                indices = random.sample(range(len(remaining_data)), num_pair - 50000)
                indices.sort()  # 保持原始顺序
                second_part = [remaining_data[i] for i in indices]
            else:
                second_part = remaining_data  # 如果剩余不够，就全取
                
            contrastive_data = first_part + second_part
            print(f"选取前50000个样本，并从剩余中随机选择{len(second_part)}个，共{len(contrastive_data)}个对比学习样本")
        else:
            # 直接取前num_pair个
            contrastive_data = contrastive_data[:num_pair]
            print(f"直接选取前{len(contrastive_data)}个对比学习样本")

    
    if num_mic is not None and len(mic_data_list) > num_mic:
        indices = random.sample(range(len(mic_data_list)), num_mic)
        indices.sort()  # 保持原始顺序
        mic_data_list = [mic_data_list[i] for i in indices]
        print(f"随机选择 {len(mic_data_list)} 个 MIC 数据样本")
    
    # 划分训练集和验证集
    contrastive_val_size = int(len(contrastive_data) * val_ratio)
    mic_val_size = int(len(mic_data_list) * val_ratio)
    
    # 随机选择验证集索引
    contrastive_val_indices = random.sample(range(len(contrastive_data)), contrastive_val_size)
    mic_val_indices = random.sample(range(len(mic_data_list)), mic_val_size)
    
    # 创建训练集和验证集的掩码
    contrastive_val_mask = [i in contrastive_val_indices for i in range(len(contrastive_data))]
    mic_val_mask = [i in mic_val_indices for i in range(len(mic_data_list))]
    
    # 划分数据
    train_contrastive_data = [item for i, item in enumerate(contrastive_data) if not contrastive_val_mask[i]]
    val_contrastive_data = [item for i, item in enumerate(contrastive_data) if contrastive_val_mask[i]]
    
    train_mic_data = [item for i, item in enumerate(mic_data_list) if not mic_val_mask[i]]
    val_mic_data = [item for i, item in enumerate(mic_data_list) if mic_val_mask[i]]
    
    # 解包数据以符合原函数的返回格式
    train_contrastive_sequences = [(seq1, seq2) for seq1, seq2, _ in train_contrastive_data]
    train_contrastive_labels = [label for _, _, label in train_contrastive_data]
    
    val_contrastive_sequences = [(seq1, seq2) for seq1, seq2, _ in val_contrastive_data]
    val_contrastive_labels = [label for _, _, label in val_contrastive_data]
    
    train_mic_sequences = [seq for seq, _ in train_mic_data]
    train_mic_labels = [label for _, label in train_mic_data]
    
    val_mic_sequences = [seq for seq, _ in val_mic_data]
    val_mic_labels = [label for _, label in val_mic_data]
    
    print(f"训练集: {len(train_contrastive_sequences)} 对比学习样本, {len(train_mic_sequences)} MIC样本")
    print(f"验证集: {len(val_contrastive_sequences)} 对比学习样本, {len(val_mic_sequences)} MIC样本")
    
    return (
        train_contrastive_sequences, train_contrastive_labels,
        val_contrastive_sequences, val_contrastive_labels,
        train_mic_sequences, train_mic_labels,
        val_mic_sequences, val_mic_labels
    )



class ContrastivePeptideDataset(Dataset):
    def __init__(self, sequence_pairs, labels, tokenizer, structure_features_dir, max_length=40):
        self.sequence_pairs = sequence_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.structure_features_dir = structure_features_dir
        
        # 预先加载所有结构特征
        self.structure_features = self._load_all_structure_features()
        
    def __len__(self):
        return len(self.sequence_pairs)
    
    def __getitem__(self, idx):
        seq1, seq2 = self.sequence_pairs[idx]
        label = self.labels[idx]
        
        # 处理序列数据
        encoding1 = self.tokenizer(seq1, return_tensors="pt", padding="max_length", 
                                  max_length=self.max_length, truncation=True)
        encoding2 = self.tokenizer(seq2, return_tensors="pt", padding="max_length", 
                                  max_length=self.max_length, truncation=True)
        
        # 获取预先加载的结构特征
        structure1 = self.structure_features[seq1]
        structure2 = self.structure_features[seq2]
        
        return {
            "input_ids1": encoding1["input_ids"].squeeze(0),
            "attention_mask1": encoding1["attention_mask"].squeeze(0),
            "structure_features1": structure1,
            "input_ids2": encoding2["input_ids"].squeeze(0),
            "attention_mask2": encoding2["attention_mask"].squeeze(0),
            "structure_features2": structure2,
            "label": torch.tensor(label, dtype=torch.float)
        }
    
    def _load_all_structure_features(self):
        structure_features = {}
        for seq1, seq2 in self.sequence_pairs:
            if seq1 not in structure_features:
                structure_features[seq1] = self._load_structure_feature(seq1)
            if seq2 not in structure_features:
                structure_features[seq2] = self._load_structure_feature(seq2)
        return structure_features
    
    def _load_structure_feature(self, seq_id):
        try:
            struct_path = f"{self.structure_features_dir}/{seq_id}_atomfeatures.npy"
            struct_feat = np.load(struct_path)
            return torch.tensor(struct_feat, dtype=torch.float)
        except Exception as e:
            print(f"Error loading structure for {seq_id}: {e}")
            return torch.zeros((self.max_length, 9), dtype=torch.float)

class MICPeptideDataset(Dataset):
    def __init__(self, sequences, mic_values, tokenizer, structure_features_dir, max_length=40):
        self.sequences = sequences
        self.mic_values = mic_values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.structure_features_dir = structure_features_dir
        
        # 预先加载所有结构特征
        self.structure_features = self._load_all_structure_features()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        mic = self.mic_values[idx]
        
        # 处理序列数据
        encoding = self.tokenizer(seq, return_tensors="pt", padding="max_length", 
                                 max_length=self.max_length, truncation=True)
        
        # 获取预先加载的结构特征
        structure = self.structure_features[seq]
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "structure_features": structure,
            "mic": torch.tensor(mic, dtype=torch.float)
        }
    
    def _load_all_structure_features(self):
        structure_features = {}
        for seq in self.sequences:
            if seq not in structure_features:
                structure_features[seq] = self._load_structure_feature(seq)
        return structure_features
    
    def _load_structure_feature(self, seq_id):
        try:
            struct_path = f"{self.structure_features_dir}/{seq_id}_atomfeatures.npy"
            struct_feat = np.load(struct_path)
            return torch.tensor(struct_feat, dtype=torch.float)
        except Exception as e:
            print(f"Error loading structure for {seq_id}: {e}")
            return torch.zeros((self.max_length, 9), dtype=torch.float)

@dataclass
class DataCollator:
    def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        processed = {}
        for key in batch[0].keys():
            if key in ["seq1", "seq2"]:
                processed[key] = [d[key] for d in batch]
            elif "input_ids" in key or "attention_mask" in key:
                processed[key] = torch.nn.utils.rnn.pad_sequence(
                    [d[key] for d in batch], batch_first=True)
            elif "structure_features" in key:
                # 对齐结构特征的长度
                processed[key] = torch.nn.utils.rnn.pad_sequence(
                    [d[key] for d in batch], batch_first=True)
            else:
                processed[key] = torch.stack([d[key] for d in batch])
        return processed
