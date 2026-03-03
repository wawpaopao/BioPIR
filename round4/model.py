# model.py 改进版
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        Args:
            query: (batch_size, seq_len, hidden_size)  # 通常是序列特征
            key_value: (batch_size, seq_len, hidden_size)  # 通常是结构特征
        Returns:
            attended_output: (batch_size, seq_len, hidden_size)
        """
        # 跨模态注意力
        attended_output, _ = self.multihead_attention(query, key_value, key_value)
        attended_output = self.dropout(attended_output)
        attended_output = self.layer_norm(query + attended_output)  # 残差连接 + LayerNorm
        return attended_output
    
class MultiTaskPeptideModel(nn.Module):
    def __init__(self, esm_model, projection_dim=256, dropout=0.1, structure_feature_dim=9,freeze_base_model=False):
        super(MultiTaskPeptideModel, self).__init__()
        
        # ESM模型作为基础编码器
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.esm_model = esm_model
        if freeze_base_model:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # 获取ESM模型的输出维度
        self.hidden_size = self.esm_model.config.hidden_size
        
        # 结构特征处理
        self.structure_encoder = nn.Linear(structure_feature_dim, self.hidden_size)
        
        # 跨模态注意力
        self.cross_attention = CrossAttention(self.hidden_size)
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, projection_dim)
        )
        
        # MIC预测头
        self.mic_prediction_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
    def encode_sequence(self, input_ids, attention_mask, structure_features):
        # 编码序列
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_embedding = outputs.last_hidden_state  # 使用[CLS]标记的表示
       
        # 编码结构特征
        encoded_structure = self.structure_encoder(structure_features)
        
        # 跨模态融合
        
        fused_embedding = self.cross_attention(sequence_embedding, encoded_structure)
        fused_embedding = fused_embedding.mean(dim=1) + sequence_embedding.mean(dim=1)
        
        return fused_embedding
    
    def forward_contrastive(self, input_ids1, attention_mask1, structure_features1, 
                           input_ids2, attention_mask2, structure_features2):
        # 编码第一个序列
        embedding1 = self.encode_sequence(input_ids1, attention_mask1, structure_features1)
        
        # 编码第二个序列
        embedding2 = self.encode_sequence(input_ids2, attention_mask2, structure_features2)
        
        # 投影到对比学习空间
        projection1 = self.projection_head(embedding1)
        projection2 = self.projection_head(embedding2)
        
        return projection1, projection2
    
    def forward_mic(self, input_ids, attention_mask, structure_features):
        # 编码序列
        embedding = self.encode_sequence(input_ids, attention_mask, structure_features)
        
        ###为了对比学习的perdict
        projection = self.projection_head(embedding)
        return projection
        # 预测MIC值
        mic_prediction = self.mic_prediction_head(embedding)
        
        return mic_prediction
    
    def forward(self, batch, task="contrastive"):
        if task == "contrastive":
            return self.forward_contrastive(
                batch["input_ids1"], batch["attention_mask1"], batch["structure_features1"],
                batch["input_ids2"], batch["attention_mask2"], batch["structure_features2"]
            )
        
        elif task == "inbatch_contrastive":
            # 新增的标准对比学习
            return self.forward_inbatch_contrastive(
                batch["input_ids"], batch["attention_mask"], batch["structure_features"]
            )
        elif task == "mic":
            
            return self.forward_mic(
                batch["input_ids"], batch["attention_mask"], batch["structure_features"]
            )
        
        else:
            raise ValueError(f"Unknown task: {task}")


    def forward_inbatch_contrastive(self, input_ids, attention_mask, structure_features):
       
        batch_size = input_ids.size(0)
        
        # 第一次前向传播，获取第一组嵌入
        embeddings_view1 = self.encode_sequence(input_ids, attention_mask, structure_features)
        projections_view1 = self.projection_head(embeddings_view1)
        projections_view1 = F.normalize(projections_view1, p=2, dim=1)
        
        # 第二次前向传播，由于dropout的随机性，会产生不同的嵌入
        embeddings_view2 = self.encode_sequence(input_ids, attention_mask, structure_features)
        projections_view2 = self.projection_head(embeddings_view2)
        projections_view2 = F.normalize(projections_view2, p=2, dim=1)
        
        # 将两组嵌入拼接起来，形成2N大小的批次
        projections_combined = torch.cat([projections_view1, projections_view2], dim=0)
        
        # 计算所有样本之间的相似度矩阵 (2N×2N)
        similarity_matrix = torch.matmul(projections_combined, projections_combined.T) / self.temperature
        
        # 创建标签：对于索引i，其正样本索引是(i+batch_size)%（2*batch_size）
        labels = torch.arange(2 * batch_size, device=input_ids.device)
        labels = (labels + batch_size) % (2 * batch_size)
        
        # 创建掩码排除自身对比
        mask = torch.ones_like(similarity_matrix) - torch.eye(2 * batch_size, device=input_ids.device)
        similarity_matrix_masked = similarity_matrix * mask
        
        # 对数值稳定性，减去每行最大值（排除自身）
        logits_max, _ = torch.max(similarity_matrix_masked, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - logits_max.detach()
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 获取每个样本对应的正样本的相似度
        pos_sim = torch.gather(similarity_matrix, 1, labels.view(-1, 1))
        
        # 计算分母（所有exp和，包括正样本，但排除自身）
        exp_sum = exp_sim.sum(dim=1, keepdim=True) - torch.exp(torch.diag(similarity_matrix).view(-1, 1))
        
        # 计算对比损失
        loss = -torch.mean(pos_sim - torch.log(exp_sum))
        
        return loss, projections_view1  # 返回第一视角的投影向量作为特征表示




class AdaptiveContrastiveLoss(nn.Module):
    def __init__(self, initial_margin=1.0, hard_weight=5.0, margin_update_factor=0.02):
        super().__init__()
        self.margin = initial_margin
        self.hard_weight = hard_weight
        self.margin_update_factor = margin_update_factor
        self.distance_history = []
        
    def forward(self, emb1, emb2, labels):
        distances = F.pairwise_distance(emb1, emb2)
        
        # 记录正负样本的距离分布
        pos_distances = distances[labels > 0.5].detach().cpu().numpy()
        neg_distances = distances[labels < 0.5].detach().cpu().numpy()
        
        if len(pos_distances) > 0 and len(neg_distances) > 0:
            self.distance_history.append((pos_distances.mean(), neg_distances.mean()))
            pos_mean = pos_distances.mean()
            neg_mean = neg_distances.mean()
            self.pos_mean=pos_mean
            self.neg_mean=neg_mean
            # 自适应调整margin
            if len(self.distance_history) >= 10:
                
                # 如果正负样本距离差距太小，增加margin
                if neg_mean - pos_mean < 0.2:
                    self.margin += self.margin_update_factor
                # 如果差距已经足够大，可以减小margin
                elif neg_mean - pos_mean > 0.3:
                    self.margin = max(0.1, self.margin - self.margin_update_factor)
        
        # 计算损失
        pos_loss = labels * distances.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        # 识别难负样本并给予更高权重
        hard_mask = (1 - labels) * (distances < self.margin).float()
        hard_loss = hard_mask * F.relu(self.margin - distances).pow(2)
        
        return pos_loss.mean() + neg_loss.mean() + self.hard_weight * hard_loss.mean()
    
    def get_stats(self):
        if len(self.distance_history) == 0:
            return {"margin": self.margin, "pos_dist": 0, "neg_dist": 0}
    
        
        return {
            "margin": self.margin,
            "pos_dist": self.pos_mean,
            "neg_dist": self.neg_mean,
            "dist_diff": self.neg_mean - self.pos_mean
        }
