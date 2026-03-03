
# BioPIR: AMP ESM Experiments (Round1-Round4)

This repository contains four rounds of experiments around ESM fine-tuning, embedding-based flow matching, and contrastive learning for peptides.

## Clone this repository

```bash
git clone https://github.com/wawpaopao/BioPIR.git
cd BioPIR
```

## Quick Setup

```bash
# Create conda environment
conda create -n esm python=3.8 -y

# Activate environment
conda activate esm

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    numpy==1.24.3 pandas==2.0.3 h5py==3.11.0 torch==2.0.1 einops==0.6.1 \
    accelerate==0.22.0 transformers==4.33.0 scikit-learn==1.3.0
```

## Round1 - ESM Fine-tuning (Regression)

- Goal: fine-tune ESM for MIC regression.
- Entry: `round1/main.py`, `round1/train.sh`.

## Round2 - ESM Embedding + Flow Matching

- Goal: generate ESM embeddings and train a flow matching model.
- Entry: `round2/preprocess_embedding.py`, `round2/main.py`.

## Round3 - Self-supervised Contrastive Learning

- Goal: self-supervised contrastive training with ESM.
- Entry: `round3/unsup_fold1.py`, `round3/train_unsup.sh`.

## Round4 - Pair Contrastive Learning + ESMFold Features

- Goal: pairwise contrastive learning with ESMFold structural features.
- Entry: `round4/main.py`, `round4/train.sh`.
