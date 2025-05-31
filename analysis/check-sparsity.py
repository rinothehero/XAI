# check_sparsity.py

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

# 모델 경로 (프루닝 + 파인튜닝 완료된 경로)
model_dir = "./pruned_bert_agnews_structured-30-3"

# 모델 로드
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)

# Linear 레이어 sparsity 확인
print("\n[Pruning Sparsity Check]")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        weight = module.weight.detach().cpu().numpy()
        row_norms = np.linalg.norm(weight, ord=1, axis=1)
        zero_rows = np.sum(row_norms == 0)
        total_rows = weight.shape[0]
        sparsity = zero_rows / total_rows * 100
        print(f"{name:50s} | sparsity: {sparsity:5.2f}%")
