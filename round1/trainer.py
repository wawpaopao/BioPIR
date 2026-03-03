
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score


class RegressionTrainer:
    def __init__(
        self,
        model,
        train_datasets,   # {'reg': train_dataset}
        val_datasets,     # {'reg': val_dataset}
        collate_fn,
        config=None
    ):
        self.config = config or {}
        self.model = model
        self.accelerator = Accelerator()
        self.collate_fn = collate_fn

        # dataloaders
        self.setup_dataloaders(train_datasets, val_datasets)

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        # loss
        self.criterion = nn.MSELoss()

        # grad accumulation
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)

        # scheduler
        total_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = total_steps_per_epoch * self.config.get('num_epochs', 10)
        num_warmup_steps = int(num_training_steps * self.config.get('warmup_ratio', 0.05))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # best ckpt by R2
        self.best_r2 = float('-inf')

        # output dir
        self.output_dir = self.config.get('output_dir', 'results')
        os.makedirs(os.path.join(self.output_dir, 'reg'), exist_ok=True)

    def setup_dataloaders(self, train_datasets, val_datasets):
        batch_size = self.config.get('batch_size', 32)
        eval_batch_size = self.config.get('eval_batch_size', 64)

        assert 'reg' in train_datasets and 'reg' in val_datasets, "datasets 必须包含键 'reg'"

        self.train_dataloader = DataLoader(
            train_datasets['reg'],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.config.get('num_workers', 4)
        )
        self.val_dataloader = DataLoader(
            val_datasets['reg'],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.config.get('num_workers', 4)
        )

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in dataloader:
            output = self.model(
                batch['input_ids'],
                batch['attention_mask'],
                task='reg'
            ).squeeze()

            labels = batch['labels']
            if isinstance(labels, (list, tuple)):
                labels = labels[0]
            labels = labels.squeeze()

            all_preds.append(output.detach().float().cpu())
            all_labels.append(labels.detach().float().cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        
        r2 = r2_score(labels, preds)
        rmse = float(np.sqrt(mean_squared_error(labels, preds)))

        order = np.argsort(preds)
        top_metrics = {}
        for k in [50, 100, 200,400,600]:
            kk = min(k, len(preds))
            if kk > 0:
                idx = order[:kk]
                top_mse = float(mean_squared_error(labels[idx], preds[idx]))
            else:
                top_mse = float('nan')
            top_metrics[f'mse_top{k}'] = top_mse

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import os

            os.makedirs(os.path.join(self.output_dir, 'reg'), exist_ok=True)
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(labels, preds, s=8, alpha=0.6, edgecolors='none')
            lo = float(min(labels.min(), preds.min()))
            hi = float(max(labels.max(), preds.max()))
            plt.plot([lo, hi], [lo, hi], 'r--', linewidth=1.0)
            plt.xlabel("True MIC")
            plt.ylabel("Predicted MIC")
            plt.title(f"Pred vs True (R2={r2:.3f}, RMSE={rmse:.3f})")
            plt.tight_layout()
            out_path = "generate_data_scatter_plot.png"
            plt.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"散点图已保存: {out_path}")
        except Exception as e:
            print(f"绘图失败（不影响训练）：{e}")

        

        return {'r2': r2, 'rmse': rmse, **top_metrics}

    @torch.no_grad()
    def predict(self, dataloader, return_numpy=True, with_labels=False, save_path=None):
        self.model.eval()
        all_preds = []
        all_labels = []  
        for batch in dataloader:
            output = self.model(
                batch['input_ids'],
                batch['attention_mask'],
                task='reg'
            ).squeeze()
            all_preds.append(output.detach().float().cpu())

            if with_labels and ('labels' in batch):
                labels = batch['labels']
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                labels = labels.squeeze()
                all_labels.append(labels.detach().float().cpu())

        preds = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels) if (with_labels and len(all_labels) > 0) else None

        if save_path is not None:
            try:
                import pandas as pd
                df = pd.DataFrame({'pred': preds.numpy()})
                if labels_tensor is not None:
                    df['label'] = labels_tensor.numpy()
                os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
                df.to_csv(save_path, index=False)
                print(f"预测结果已保存: {save_path}")
            except Exception as e:
                print(f"保存预测结果失败：{e}")

        if return_numpy:
            preds_out = preds.numpy()
            labels_out = labels_tensor.numpy() if labels_tensor is not None else None
        else:
            preds_out = preds
            labels_out = labels_tensor

        return preds_out, labels_out

    def save_checkpoint(self, epoch, metrics):
        if metrics['r2'] > self.best_r2:
            self.best_r2 = metrics['r2']
            save_path = os.path.join(self.output_dir, 'reg', 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
                'metrics': metrics,
            }, save_path)

    def train(self, num_epochs):
        prepared = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader
        )
        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.train_dataloader = prepared[2]
        self.val_dataloader = prepared[3]

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch, num_epochs)
            val_metrics = self.evaluate(self.val_dataloader)
            # 汇总并打印 TopK MSE 指标
            top_keys = sorted([k for k in val_metrics.keys() if k.startswith('mse_top')], key=lambda x: int(x.replace('mse_top','')))
            top_str = '  '.join([f"{k}:{val_metrics[k]:.4f}" for k in top_keys]) if top_keys else ''
            print(f"\nEpoch {epoch+1}/{num_epochs}  TrainLoss: {train_loss:.4f}  Val R2: {val_metrics['r2']:.4f}  Val RMSE: {val_metrics['rmse']:.4f}  {top_str}")
            self.save_checkpoint(epoch, val_metrics)

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        total_steps = len(self.train_dataloader)
        progress_bar = tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        self.optimizer.zero_grad()
        for step, batch in enumerate(self.train_dataloader):
            
            output = self.model(
                batch['input_ids'],
                batch['attention_mask'],
                task='reg'
            ).squeeze()

            labels = batch['labels']
            if isinstance(labels, (list, tuple)):
                labels = labels[0]
            labels = labels.squeeze()
        
            loss = self.criterion(output, labels) / self.gradient_accumulation_steps
            self.accelerator.backward(loss)

            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1 == total_steps):
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({'loss': total_loss / (step + 1), 'lr': current_lr})
            progress_bar.update(1)

        progress_bar.close()
        return total_loss / total_steps
        