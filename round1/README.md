# Round1 - ESM Fine-tuning

This folder fine-tunes ESM for MIC regression.

## Key Steps
1. Update dataset paths in `dataset.py` (`reg_train_and_generate_data.csv`, `reg_val.csv`).
2. Set `MODEL_NAME_OR_PATH` in `main.py` to your local ESM model path.
3. Adjust training hyperparameters and output directory in `config.yaml`.
4. Launch training:

```bash
bash train.sh
```
