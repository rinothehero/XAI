from .shrinking_utils import apply_structural_shrink
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


if __name__ == "__main__":
    model_path = "models/pruned/bert_agnews_20pct_v1" # Placeholder
    save_path = "models/shrunk/bert_agnews_pruned_20_shrunk_dir"

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    shrunk_model = apply_structural_shrink(model)

    shrunk_model.save_pretrained(save_path)
    import os; os.makedirs(save_path, exist_ok=True); print(f"✅ 구조 축소 완료 및 저장: {save_path}")
