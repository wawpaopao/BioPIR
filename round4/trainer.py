# trainer.py 改进版
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from dataset import  DataCollator
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from model import AdaptiveContrastiveLoss
from torch.utils.data import RandomSampler

class ContrastiveTrainer:
    def __init__(self, model, mic_train_dataset,contrastive_train_dataset,contrastive_loss_fn,mic_loss_fn, mic_val_dataset=None,contrastive_val_dataset=None, config=None):
        self.config = config or {}
        self.model = model
        self.train_mic_dataset = mic_train_dataset
        self.train_contrastive_dataset = contrastive_train_dataset
        self.val_mic_dataset = mic_val_dataset
        self.val_contrastive_dataset = contrastive_val_dataset
        self.accelerator = Accelerator()
        self.contrastive_loss_fn = contrastive_loss_fn
        self.mic_loss_fn = mic_loss_fn
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # 跟踪指标
        self.train_metrics = {
            'loss': [], 'pos_dist': [], 'neg_dist': [], 'margin': [],'mic_loss': []
        }
        self.val_metrics = {
            'contrastive_loss': [], 'auc': [],'mic_loss':[],'r2':[],'rmse':[]
        }
        
        # 是否使用wandb
        self.use_wandb = self.config.get('use_wandb', False)
    
    def train_epoch(self, epoch, num_epochs, contrastive_loader, 
                    mic_loader, optimizer, contrastive_loss_fn, 
                    mic_loss_fn, accelerator,scheduler,
                     contrastive_weight=0.1, accumulation_steps=2):
        self.model.train()
        total_loss = 0
        epoch_pos_dist = 0
        epoch_neg_dist = 0
        epoch_margin = 0
        epoch_bc_loss = 0

        contrastive_steps = len(contrastive_loader)
        mic_steps = len(mic_loader)
        num_batches = 0
        contrastive_iter = iter(contrastive_loader)
        mic_iter = iter(mic_loader)

        total_steps = max(contrastive_steps, mic_steps)

        progress_bar = tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{num_epochs}")

        optimizer.zero_grad()  # 初始化梯度

        for step in range(total_steps):
            step_loss = 0
            contrastive_loss_stats = {'pos_dist': 0, 'neg_dist': 0, 'margin': 0}
            inbatch_contrastive_loss_value = 0

            if step < contrastive_steps and contrastive_weight > 0:  # 仅在contrastive_weight > 0时计算
                try:
                    contrastive_batch = next(contrastive_iter)
                except StopIteration:
                    contrastive_iter = iter(contrastive_loader)
                    contrastive_batch = next(contrastive_iter)

                proj1, proj2 = self.model(contrastive_batch, task="contrastive")
                contrastive_loss = contrastive_loss_fn(proj1, proj2, contrastive_batch["label"])
                step_loss += contrastive_weight * contrastive_loss
                contrastive_loss_stats = contrastive_loss_fn.get_stats()

            if step < mic_steps and contrastive_weight < 1:
                try:
                    mic_batch = next(mic_iter)
                except StopIteration:
                    mic_iter = iter(mic_loader)
                    mic_batch = next(mic_iter)

                inbatch_contrastive_loss,_ = self.model(mic_batch, task="inbatch_contrastive")
        
                step_loss += (1 - contrastive_weight) * inbatch_contrastive_loss
                inbatch_contrastive_loss_value = inbatch_contrastive_loss.item()
                
            # 使用accelerator处理反向传播
            accelerator.backward(step_loss / accumulation_steps)  # 梯度累积，除以累积步数

            # 每 accumulation_steps 步更新一次模型
            if (step + 1) % accumulation_steps == 0 or step == total_steps - 1:
                optimizer.step()
                optimizer.zero_grad()
                
                scheduler.step()  # 更新学习率
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            progress_bar.set_postfix({
                'loss': step_loss.item(),
                'pos_dist': contrastive_loss_stats['pos_dist'],
                'neg_dist': contrastive_loss_stats['neg_dist'],
                'margin': contrastive_loss_stats['margin'],
                'bc_loss': inbatch_contrastive_loss_value,
                'lr': current_lr  # 添加学习率到进度条
            })
            progress_bar.update(1)

            total_loss += step_loss.item()
            epoch_pos_dist += contrastive_loss_stats['pos_dist']
            epoch_neg_dist += contrastive_loss_stats['neg_dist']
            epoch_margin += contrastive_loss_stats['margin']
            epoch_bc_loss += inbatch_contrastive_loss_value
            num_batches += 1

        progress_bar.close()

        metrics = {
            'loss': total_loss / num_batches,
            'pos_dist': epoch_pos_dist / num_batches,
            'neg_dist': epoch_neg_dist / num_batches,
            'margin': epoch_margin / num_batches,
            'mic_loss': epoch_bc_loss / num_batches
        }

        return metrics
    def train(self, epochs=None, contrastive_weight=0.1):
        """
        训练模型
        参数:
            epochs: 训练的轮数，如果为None则使用config中的设置
            contrastive_weight: 对比学习损失的权重
        """
        # 使用传入的epochs或配置中的默认值
        collate_function = DataCollator()
        num_epochs = epochs if epochs is not None else self.config.get('num_epochs', 10)
        
        # 创建数据加载器
        train_contrastive_dataloader = DataLoader(
            self.train_contrastive_dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            collate_fn=collate_function,
            num_workers=self.config.get('num_workers', 4)
        )
        
        train_mic_dataloader = DataLoader(
            self.train_mic_dataset, 
            batch_size=2*self.config.get('batch_size', 32),
            sampler=RandomSampler(
                self.train_mic_dataset, 
                replacement=True,  # 允许重复采样
                num_samples=2*len(self.train_mic_dataset)  # 采样总次数=数据集大小×4
            ),
            
            collate_fn=collate_function,
            num_workers=self.config.get('num_workers', 4)
        )
                
        # 创建验证集数据加载器
        val_contrastive_dataloader = None
        val_mic_dataloader = None
        
        # if self.val_contrastive_dataset:
        #     val_contrastive_dataloader = DataLoader(
        #         self.val_contrastive_dataset,
        #         batch_size=self.config.get('eval_batch_size', 64),
        #         collate_fn=collate_function,
        #         num_workers=self.config.get('num_workers', 4)
        #     )
            
        # if self.val_mic_dataset:
        #     val_mic_dataloader = DataLoader(
        #         self.val_mic_dataset,
        #         batch_size=self.config.get('eval_batch_size', 64),
        #         collate_fn=collate_function,
        #         num_workers=self.config.get('num_workers', 4)
        #     )
        
        # 设置学习率调度器
        total_steps = len(train_contrastive_dataloader) * num_epochs
        warmup_steps = int(total_steps * 0.02)
        
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 使用accelerator准备模型、优化器和数据加载器
        self.model, self.optimizer, train_contrastive_dataloader, train_mic_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_contrastive_dataloader, train_mic_dataloader
        )
        
        if val_contrastive_dataloader:
            val_contrastive_dataloader = self.accelerator.prepare(val_contrastive_dataloader)
        
        if val_mic_dataloader:
            val_mic_dataloader = self.accelerator.prepare(val_mic_dataloader)
        
        # 初始化wandb
        if self.use_wandb:
            wandb.init(
                project=self.config.get('wandb_project', 'peptide-contrastive'),
                name=self.config.get('wandb_run_name', None),
                config=self.config
            )
        
        # 训练循环
        best_val_metric = float('-inf')  # 用于模型选择
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_metrics = self.train_epoch(
                epoch=epoch, 
                num_epochs=num_epochs,
                contrastive_loader=train_contrastive_dataloader, 
                mic_loader=train_mic_dataloader, 
                optimizer=self.optimizer, 
                scheduler=scheduler,
                contrastive_loss_fn=self.contrastive_loss_fn, 
                mic_loss_fn=self.mic_loss_fn, 
                accelerator=self.accelerator,
                contrastive_weight=contrastive_weight
            )
            
            
            # 记录训练指标
            self.train_metrics['loss'].append(train_metrics['loss'])
            self.train_metrics['pos_dist'].append(train_metrics['pos_dist'])
            self.train_metrics['neg_dist'].append(train_metrics['neg_dist'])
            self.train_metrics['margin'].append(train_metrics['margin'])
            self.train_metrics['mic_loss'].append(train_metrics['mic_loss'])
            
            # 验证
            if val_contrastive_dataloader and val_mic_dataloader:
                val_metrics = self.evaluate(val_contrastive_dataloader, val_mic_dataloader)
                
                self.val_metrics['contrastive_loss'].append(val_metrics['contrastive_loss'])
                self.val_metrics['auc'].append(val_metrics['auc'])
                self.val_metrics['mic_loss'].append(val_metrics['mic_loss'])
                self.val_metrics['r2'].append(val_metrics['r2'])
                self.val_metrics['rmse'].append(val_metrics['rmse'])
                
                # 打印验证结果
                print(f"Epoch {epoch+1}/{num_epochs} - Validation: AUC={val_metrics['auc']:.4f}, R2={val_metrics['r2']:.4f}, RMSE={val_metrics['rmse']:.4f}")
                
                # 记录到wandb
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_metrics['loss'],
                        "train_pos_dist": train_metrics['pos_dist'],
                        "train_neg_dist": train_metrics['neg_dist'],
                        "train_margin": train_metrics['margin'],
                        "train_mic_loss": train_metrics['mic_loss'],
                        "val_contrastive_loss": val_metrics['contrastive_loss'],
                        "val_auc": val_metrics['auc'],
                        "val_mic_loss": val_metrics['mic_loss'],
                        "val_r2": val_metrics['r2'],
                        "val_rmse": val_metrics['rmse'],
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
                
                # 保存最佳模型
                # 这里使用AUC和R2的平均值作为模型选择的指标
                current_val_metric = (val_metrics['auc'] + val_metrics['r2']) / 2
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    self.save_model(self.config.get('output_dir', 'results'), f"best_model_epoch_{epoch+1}")
            else:
                # 如果没有验证集，只记录训练指标
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_metrics['loss'],
                        "train_pos_dist": train_metrics['pos_dist'],
                        "train_neg_dist": train_metrics['neg_dist'],
                        "train_margin": train_metrics['margin'],
                        "train_mic_loss": train_metrics['mic_loss'],
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
            
            # 每个epoch结束后保存模型
            if (epoch + 1) % self.config.get('save_every', 2) == 0:
                self.save_model(self.config.get('output_dir', 'results'), f"model_epoch_{epoch+1}")
        
        # 训练结束后保存最终模型
        self.save_model(self.config.get('output_dir', 'results'), "final_model")
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 关闭wandb
        if self.use_wandb:
            wandb.finish()
    
    def save_model(self, output_dir, model_name):
        """保存模型"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 获取未包装的模型
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # 保存模型状态
        torch.save(unwrapped_model.state_dict(), os.path.join(model_dir, "model.pt"))
        
        # 如果使用ESM模型，也可以保存投影层
        if hasattr(unwrapped_model, 'projector'):
            torch.save(unwrapped_model.projector.state_dict(), os.path.join(model_dir, "projector.pt"))
        
        print(f"Model saved to {model_dir}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # 绘制对比学习损失曲线
        axs[0, 0].plot(self.train_metrics['loss'], label='Train Loss')
        if self.val_metrics['contrastive_loss']:
            axs[0, 0].plot(self.val_metrics['contrastive_loss'], label='Val Contrastive Loss')
        axs[0, 0].set_title('Contrastive Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        
        # 绘制MIC损失曲线
        axs[0, 1].plot(self.train_metrics['mic_loss'], label='Train MIC Loss')
        if self.val_metrics['mic_loss']:
            axs[0, 1].plot(self.val_metrics['mic_loss'], label='Val MIC Loss')
        axs[0, 1].set_title('MIC Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        
        # 绘制距离曲线
        axs[0, 2].plot(self.train_metrics['pos_dist'], label='Positive Distance')
        axs[0, 2].plot(self.train_metrics['neg_dist'], label='Negative Distance')
        axs[0, 2].plot(self.train_metrics['margin'], label='Margin')
        axs[0, 2].set_title('Distances and Margin')
        axs[0, 2].set_xlabel('Epoch')
        axs[0, 2].set_ylabel('Distance')
        axs[0, 2].legend()
        
        # 绘制AUC曲线
        if self.val_metrics['auc']:
            axs[1, 0].plot(self.val_metrics['auc'])
            axs[1, 0].set_title('Validation AUC')
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].set_ylabel('AUC')
        
        # 绘制R2曲线
        if self.val_metrics['r2']:
            axs[1, 1].plot(self.val_metrics['r2'])
            axs[1, 1].set_title('Validation R²')
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('R²')
        
        # 绘制RMSE曲线
        if self.val_metrics['rmse']:
            axs[1, 2].plot(self.val_metrics['rmse'])
            axs[1, 2].set_title('Validation RMSE')
            axs[1, 2].set_xlabel('Epoch')
            axs[1, 2].set_ylabel('RMSE')
        
        # 保存图表
        plt.tight_layout()
        os.makedirs(self.config.get('output_dir', 'results'), exist_ok=True)
        plt.savefig(os.path.join(self.config.get('output_dir', 'results'), 'training_curves.png'))
        
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(plt)})
    
    def evaluate(self, contrastive_dataloader, mic_dataloader):
        """
        评估模型在验证集上的性能
        
        参数:
            contrastive_dataloader: 对比学习任务的数据加载器
            mic_dataloader: MIC回归任务的数据加载器
            
        返回:
            包含各种评估指标的字典
        """
        self.model.eval()
        
        # 评估对比学习任务
        total_contrastive_loss = 0
        total_pos_dist = 0
        total_neg_dist = 0
        total_margin = 0
        all_labels = []
        all_scores = []
        contrastive_batches = 0

        with torch.no_grad():
            for batch in tqdm(contrastive_dataloader, desc="Evaluating contrastive"):
                # 前向传播
                proj1, proj2 = self.model(batch, task="contrastive")
                
                # 计算损失
                contrastive_loss = self.contrastive_loss_fn(proj1, proj2, batch['label'])
                loss_stats = self.contrastive_loss_fn.get_stats()

                # 累积损失和距离
                total_contrastive_loss += contrastive_loss.item()
                total_pos_dist += loss_stats['pos_dist']
                total_neg_dist += loss_stats['neg_dist']
                total_margin += loss_stats['margin']
                contrastive_batches += 1

                # 计算相似度得分
                distances = F.pairwise_distance(proj1, proj2)
                scores = -distances  # 距离越小，得分越高
                all_labels.extend(batch['label'].cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

        # 评估MIC回归任务
        total_mic_loss = 0
        all_preds = []
        all_targets = []
        mic_batches = 0

        with torch.no_grad():
            for batch in tqdm(mic_dataloader, desc="Evaluating MIC"):
                # 前向传播
                logits = self.model(batch, task="mic")
                
                # 计算损失
                mic_loss = self.mic_loss_fn(logits, batch['mic'].unsqueeze(1))
                

                # 累积损失
                total_mic_loss += mic_loss.item()
                mic_batches += 1
                
                # 收集预测和目标值
                all_preds.extend(logits.squeeze().cpu().numpy())
                all_targets.extend(batch['mic'].cpu().numpy())
     
        # 计算平均损失和距离
        avg_contrastive_loss = total_contrastive_loss / max(contrastive_batches, 1)
        avg_pos_dist = total_pos_dist / max(contrastive_batches, 1)
        avg_neg_dist = total_neg_dist / max(contrastive_batches, 1)
        avg_margin = total_margin / max(contrastive_batches, 1)
        avg_mic_loss = total_mic_loss / max(mic_batches, 1)

        # 计算AUC
        auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5
        
        # 计算回归指标
        r2 = r2_score(all_targets, all_preds) if all_targets and all_preds else 0
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds)) if all_targets and all_preds else 0

        return {
            'contrastive_loss': avg_contrastive_loss,
            'pos_dist': avg_pos_dist,
            'neg_dist': avg_neg_dist,
            'margin': avg_margin,
            'auc': auc,
            'mic_loss': avg_mic_loss,
            'r2': r2,
            'rmse': rmse
        }