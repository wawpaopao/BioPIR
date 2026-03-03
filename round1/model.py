# model.py 改进版
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np

class RegressionModel(nn.Module):
    def __init__(self, esm_model, projection_dim=256, dropout=0.1, freeze_base_model=False):
        super(RegressionModel, self).__init__()
        
        self.esm_model = esm_model
        if freeze_base_model:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.esm_model.config.hidden_size
        
        self.amp_reg_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )

    
    def forward(self, input_ids, attention_mask, task="amp_cls"):
        
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_embedding = outputs.last_hidden_state[:,0,:]

        reg_output = self.amp_reg_head(sequence_embedding)
        return reg_output







