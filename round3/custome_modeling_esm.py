import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmPreTrainedModel, EsmModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Tuple
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

class EsmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        #config.hidden_size = 128
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        #x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features.mean(dim=1)  # 取平均值
        x = self.dropout(x)
        x = self.dense(x)
        encode_embedding = x
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x,encode_embedding

class Unsup_regression(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.esm = EsmModel(config, add_pooling_layer=False)
        #self.encoder = nn.Linear(config.hidden_size, 128)
        self.classifier = EsmClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,  # 修改为FloatTensor以适应回归任务
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
         # 取平均值
        #encoded_embedding = self.encoder(mean_embedding)

        #
        #logits = self.classifier(encoded_embedding)
        logits,contrastive_embedding = self.classifier(sequence_output)
        # return logits, sequence_output.mean(dim=1)
        return logits, contrastive_embedding
        contrastive_loss = simcse_unsup_loss(contrastive_embedding, temp=0.05)
        return SequenceClassifierOutput(
             loss=contrastive_loss,
             logits=logits,
             hidden_states=outputs.hidden_states,
             attentions=outputs.attentions,
         )
        
        regression_loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            regression_loss = F.mse_loss(logits.squeeze(), labels)

       
        alpha = 1
        beta = 0
        loss = None
        if contrastive_loss is not None and regression_loss is not None:
            loss = alpha*contrastive_loss + beta*regression_loss
        
        
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




def simcse_unsup_loss(y_pred, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): 模型的输出特征, [batch_size * 2, dim]
    temp (float): 温度系数，用于控制相似度的缩放
    """
    device = y_pred.device
    # 得到y_pred对应的标签, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    
    # 计算余弦相似度矩阵
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    
    # 将相似度矩阵对角线置为很小的值，消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    
    # 相似度矩阵除以温度系数
    sim = sim / temp
    
    # 计算交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    
    return torch.mean(loss)


