import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math
from pathlib import Path
from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm
from itertools import cycle
import torch.nn as nn
import numpy as np

def divisible_by(numerator, denominator):
    return numerator % denominator == 0


class ProteinFlowTrainer(nn.Module):
    def __init__(
        self,
        rectified_flow_model,
        dataset: Dataset,
        *,
        num_train_steps: int = 100000,
        learning_rate: float = 5e-3,
        batch_size: int = 64,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 32,
        use_ema: bool = False,
        max_grad_norm: float = 1.0,
        max_len: int = 28,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        super().__init__()

        # 初始化 Accelerator
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.model = rectified_flow_model

        self.num_train_steps = num_train_steps

        self.use_ema = use_ema
        self.ema_model = None
        if self.is_main and use_ema:
            self.ema_model = EMA(self.model, beta=0.995, **ema_kwargs)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        num_warmup_steps = num_train_steps*0.05

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warm-up
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - num_warmup_steps) / float(max(1, num_train_steps - num_warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine from 1 -> 0

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        self.dl_cycle = cycle(self.dl)

        # 使用 Accelerator 准备所有组件
        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        # 文件夹设置
        self.results_folder = Path(results_folder)
        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)

        self.save_results_every = save_results_every
        self.checkpoint_every = checkpoint_every
        self.num_samples = num_samples
        self.max_grad_norm = max_grad_norm
        self.max_len = max_len
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = {
            'model': self.accelerator.unwrap_model(self.model).state_dict(),
        }
        if self.use_ema:
            save_package['ema_model'] = self.ema_model.state_dict()

        torch.save(save_package, str(path))
        print(f"Checkpoint saved to {path}")

    def load(self, path):
        path = Path(path)
        assert path.exists()
        
        load_package = torch.load(str(path), map_location=self.accelerator.device)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(load_package['model'])
        
        if self.use_ema and 'ema_model' in load_package:
            self.ema_model.load_state_dict(load_package['ema_model'])
        
        print(f"Checkpoint loaded from {path}")

    @torch.no_grad()
    def sample_and_save(self, step):
        """生成样本并保存"""
        if not self.is_main:
            return

        eval_model = self.ema_model if self.use_ema else self.accelerator.unwrap_model(self.model)
        eval_model.eval()
        target_length = np.random.randint(7,20)
       
        padding_mask = torch.ones(self.num_samples, self.max_len + 2, dtype=torch.bool)
        padding_mask[:, target_length+2:] = 0  # Mask padding positions
        x_start = torch.randn(self.num_samples, self.max_len + 2, 320)
        x_start = x_start.to(self.accelerator.device)
        padding_mask = padding_mask.to(self.accelerator.device)
        generated_embeddings = eval_model.sample(x_start=x_start,padding_mask=padding_mask)

        target_length_list = [target_length] * self.num_samples
        target_length_list = torch.tensor(target_length_list).to(self.accelerator.device)
        torch.save(target_length_list, self.results_folder / f'target_length_{step}.pt')
        save_path = self.results_folder / f'sample_{step}.pt'
        torch.save(generated_embeddings, str(save_path))
        
        self.accelerator.print(f"Generated {self.num_samples} samples and saved to {save_path}")
        
    def train(self):
        """主训练循环"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params/1000000} million")
        
        pbar = tqdm(range(self.num_train_steps), desc="Training", disable=not self.is_main)

        for step in pbar:
            self.model.train()

            batch = next(self.dl_cycle)
            
            target_embeds = batch['target']
            padding_mask = batch['padding_mask']
            
            target_embeds = target_embeds.to(self.accelerator.device)
            padding_mask = padding_mask.to(self.accelerator.device)
            
            
            loss = self.model(target_embeds, padding_mask=padding_mask)
            
            if self.is_main:
                pbar.set_postfix(loss=loss.item())

            self.accelerator.backward(loss)
            
            if self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.accelerator.wait_for_everyone()

            if self.is_main and self.use_ema:
                self.ema_model.update()

            if self.is_main:
                if divisible_by(step, self.save_results_every) and step!=0:
                    self.sample_and_save(step)
                    
                if divisible_by(step, self.checkpoint_every) and step!=0:
                    self.save(self.checkpoints_folder / f'checkpoint_{step}.pt')
            
        self.accelerator.print('Training complete')