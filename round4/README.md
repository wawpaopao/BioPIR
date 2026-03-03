# Round4 - Pair Contrastive Learning + ESMFold Features

This folder trains contrastive models on paired data and adds ESMFold structural features.

## Key Steps
1. In `config_contrastive_only.yaml`, set contrastive data, MIC data, ESMFold feature directory, and training hyperparameters.
2. In `main.py`, set `MODEL_NAME_OR_PATH` to your local ESM model path.
3. Launch training:

```bash
bash train.sh
```
