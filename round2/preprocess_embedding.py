from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer,AutoModelForMaskedLM   
import pandas as pd
# Your SequenceDataset class would go here...
class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer(
            sequence, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=30 # Example max length
        )
        return {
            'sequence': sequence,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

def get_residue_embeddings(sequences, tokenizer, model, device, batch_size=32, max_length=42):
    """
    获取每个残基（氨基酸）的嵌入表示。
    输出形状为 [num_sequences, length, dim]。
    """
    dataset = SequenceDataset(sequences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.to(device)
    model.eval()
    
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Residue Embeddings"):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # 1. 获取模型输出
            outputs = model.esm(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True
                                )
            
    
            last_hidden_state = outputs.last_hidden_state

            all_embeddings.append(last_hidden_state.cpu().numpy())
    
    # 将所有批次的结果拼接起来
    return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    # --- 1. Load your model and tokenizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME_OR_PATH = '/data/wangaw/ESM/esm_model_8M' 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, padding_side='right', use_fast=True,
                                         model_max_length=30, trust_remote_code=True)

# 加载模型
    esm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    
    # --- 2. Prepare your sequences ---
    df_sequences = pd.read_csv("/home/wangaw/ESM/data/broad_GM_MIC_regression_2k_neg.csv")
    df_sequences = df_sequences[df_sequences['label']<8]
    sequences_to_embed = df_sequences['Sequence'].tolist()
    
    # --- 3. Call the function ---
    final_embeddings = get_residue_embeddings(sequences_to_embed, tokenizer, esm_model, device)

    # --- 4. Inspect the results ---
    print(f"\nSuccessfully generated embeddings for {final_embeddings.shape[0]} sequences.")
    print(f"Shape of the final embedding matrix: {final_embeddings.shape}")
    output_file = '/data/wangaw/ESM/flow_matching/round2_data_aug/data/protein_embeddings_input_esm8million_1w.npz'

    np.savez_compressed(
        output_file, 
        embeddings=final_embeddings, 
        sequences=sequences_to_embed
    )


