# Round2 - ESM Embedding + Flow Matching

This folder trains a flow matching model based on ESM embeddings.

## Key Steps
1. In `preprocess_embedding.py`, set `MODEL_NAME_OR_PATH`, input CSV path, and `output_file`.
2. Generate embeddings:

```bash
python preprocess_embedding.py
```

3. In `main.py`, set `embedding_file`, `checkpoints_folder`, and `results_folder`.
4. Launch training:

```bash
python main.py
```
