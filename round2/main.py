from model import RectifiedFlow1D,SimpleTransformerDenoiser
from dataset import PeptideDataset
from trainer import ProteinFlowTrainer
import torch

if __name__ == '__main__':
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PeptideDataset(embedding_file='/data/wangaw/ESM/flow_matching/round2_data_aug/data/protein_embeddings_input_esm8million.npz',
                              max_len=28, file_format='npz') 


    denoise_model = SimpleTransformerDenoiser(num_layers=4)
    rectified_flow = RectifiedFlow1D(model=denoise_model,seq_length=28,timesteps=100,device=device)
    # 2. 实例化并配置 Trainer
    # --------------------------------
    trainer = ProteinFlowTrainer(
        rectified_flow_model=rectified_flow, # 传入您的 Flow 模型
        dataset=dataset,                     # 传入您的数据集
        num_train_steps=500000,              # 总训练步数
        learning_rate=2e-5,
        batch_size=128,
        checkpoint_every=5000,               # 每5000步保存一次模型
        save_results_every=5000,             # 每1000步生成一次样本
        checkpoints_folder='/data/wangaw/ESM/flow_matching/round2_data_aug/protein_checkpoints_transformer_8million_simple_4layers',
        results_folder='/data/wangaw/ESM/flow_matching/round2_data_aug/protein_samples_transformer_8million_simple_4layer',
        accelerate_kwargs={
            'mixed_precision': 'no' # 使用混合精度训练
        }
    )
    # 3. 开始训练
    # --------------------------------
    trainer.train()