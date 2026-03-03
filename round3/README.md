# Round3 - Self-supervised Contrastive Learning

This folder runs self-supervised contrastive training with ESM.

## Key Steps
1. In `unsup_fold1.py`, set `MODEL_NAME_OR_PATH` and train/val data paths.
2. Adjust hyperparameters in `train_unsup.sh` (e.g., `lr`, `batch_size`, `num_train_epochs`, `output_dir`).
3. Launch training:

```bash
bash train_unsup.sh
```
