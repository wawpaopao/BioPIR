from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import h5py

class PeptideDataset(Dataset):
    def __init__(self, embedding_file, max_len=28, file_format='npz'):
        """
        target_data: list of target peptide sequences
        embedding_file: path to pre-computed embeddings file
        max_len: maximum sequence length
        file_format: 'npz', 'pt', 'h5', or 'pkl'
        """
        self.max_len = max_len
        
        # Load pre-computed embeddings
        self.embeddings,self.sequences = self._load_embeddings(embedding_file, file_format)
        
        # Verify data consistency
        assert len(self.sequences) == len(self.embeddings), \
            f"Mismatch: {len(self.sequences)} sequences but {len(self.embeddings)} embeddings"
        
    def _load_embeddings(self, embedding_file, file_format):
        """Load embeddings based on file format"""
        if file_format == 'npz':
            data = np.load(embedding_file,allow_pickle=True)
            return data['embeddings'] ,data['sequences'] # Assuming key is 'embeddings'
        elif file_format == 'pt':
            return torch.load(embedding_file)
        elif file_format == 'h5':
            with h5py.File(embedding_file, 'r') as f:
                return f['embeddings'][:]
        elif file_format == 'pkl':
            with open(embedding_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        target_seq = self.sequences[idx]
        
        # Get pre-computed embedding
        target_embed = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        
        # Get sequence length
        seq_len = len(target_seq)
        
        # Create padding mask (True for padded positions)
        padding_mask = torch.ones(self.max_len + 2, dtype=torch.bool)
        padding_mask[seq_len+2:] = 0  # Mask padding positions
        
        return {
            'target': target_embed,
            'padding_mask': padding_mask,
            'seq_len': seq_len,
            'sequence': target_seq  # Keep original sequence for reference
        }





