import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Sequence
import torch
import transformers
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset
import pandas as pd
from custome_modeling_esm import Unsup_regression
from datetime import datetime
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for the training loop."""
    num_train_epochs: int = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)
    weight_decay: float = field(default=0.05)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={'help': 'Maximum sequence length. Sequences will be right padded (and possibly truncated).',},)
    flash_attn : Optional[bool] = field(default=False)
    output_dir: str = field(default="output")
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    seed: int = field(default=42)
    learning_rate: float = field(default=1e-4)
    #lr_scheduler_type: str = field(default="cosine_with_restarts")
    warmup_steps: int = field(default=50)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=1000)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=1)
    checkpointing: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    save_model: bool = field(default=False)
    report_to: Optional[str] = field(default='none')
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict) 

class Supervised_Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: any,
                 max_length: int = 100):
        super(Supervised_Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        data = pd.read_csv(data_path)
        data_cleaned = data.dropna(subset=['Sequence'])
        data_cleaned = data_cleaned.reset_index(drop=True)
        self.data = data_cleaned
        self.sequences = self.data['Sequence'].tolist()
        self.labels = self.data['label'].tolist()

        self.input_ids, self.attention_mask = self._tokenize_sequences()

    def _tokenize_sequences(self):
        # 为无监督对比学习，每个样本重复两次
        sequences = [seq for seq in self.sequences for _ in range(2)]
        
        # 对序列进行tokenization
        tokenized_data = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized_data["input_ids"]
        attention_mask = tokenized_data["attention_mask"]
        return input_ids, attention_mask

    def __len__(self):
        # 每个样本重复两次，所以总长度是原始数据的两倍
        return len(self.sequences) * 2

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_id = self.input_ids[i]
        attention_mask = self.attention_mask[i]
        label_index = i // 2  
        label = torch.tensor(self.labels[label_index], dtype=torch.float)

        return dict(input_ids=input_id, attention_mask=attention_mask, labels=label)

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids = torch.stack([instance["input_ids"] for instance in instances])
        labels = torch.tensor([instance["labels"] for instance in instances], dtype=torch.float)
        attention_mask = torch.stack([instance["attention_mask"] for instance in instances])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_args = parser.parse_args_into_dataclasses()[0]
    
    MODEL_NAME_OR_PATH = '/data/wangaw/ESM/esm_model'

    tokenizer =  AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH,padding_side='right',use_fast=True,
                                               model_max_length=training_args.model_max_length,
                                                trust_remote_code=True,)
    
    fold_num = 1
    train_file = "/home/wangaw/ESM/semi_supervised/code_submission/round2_semi_sup/data/reg_train_and_generate_data.csv"
    test_file = "/home/wangaw/ESM/semi_supervised/code_submission/round1/data/reg_val.csv"

    # 加载数据集
    train_dataset = Supervised_Dataset(train_file, tokenizer)
    test_dataset = Supervised_Dataset(test_file, tokenizer)
    
    model = Unsup_regression.from_pretrained(MODEL_NAME_OR_PATH,num_labels=1,trust_remote_code=True)

    
    for param in model.esm.parameters():
        param.requires_grad = True

# # 只解冻分类层的参数
    
    print('Creating and saving datasets...')
    print('Creating and saving datasets...')
    
    
    print(len(train_dataset[0]['input_ids']))
    print(train_dataset[0])
    
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())

    print(f" base model - Total size={n_params/2**20:.2f}M params")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_trainable_params}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   train_dataset=train_dataset,
                                   eval_dataset=test_dataset,
                                   data_collator=data_collator)
    trainer.train()
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)




if __name__ == "__main__":
    train()
