# run_struct_prune.py

import os
import torch
import torch.nn.utils.prune as prune
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import numpy as np

# 0. 설정
orig_model_dir = "../models/ft_BERT_agnews_full_dataset"  # Fine-tuned 모델 경로
prune_model_dir = "../pruned_bert_agnews_structured-20-1"
os.makedirs(prune_model_dir, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 모델·토크나이저 로드
model     = AutoModelForSequenceClassification.from_pretrained(orig_model_dir).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(orig_model_dir)

# 2. 구조적 Pruning 대상 수집 (모든 Linear 레이어)
to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        to_prune.append((name, module, 'weight'))

# 3. Structured Pruning 적용
for _, module, weight_name in to_prune:
    prune.ln_structured(
        module,
        name=weight_name,
        amount=0.2,   # 제거 비율: 20%
        n=1,          # L1-norm
        dim=0         # 행(row) 단위로 prune
    )

# ✅ [수정] Pruning mask를 실제 weight에 반영하고 제거 (Reparameterization 제거)
for _, module, weight_name in to_prune:
    prune.remove(module, weight_name)

for name, module, _ in to_prune:
    weight = module.weight.detach().cpu().numpy()
    row_norms = np.linalg.norm(weight, ord=1, axis=1)
    zero_rows = np.sum(row_norms == 0)
    print(f"{name} | sparsity: {zero_rows / weight.shape[0] * 100:.2f}%")


# 4. 데이터셋 로드 및 전처리 (AG News 예시)
dataset = load_dataset("ag_news")
split   = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds   = split["test"]

def tokenize_fn(ex):
    return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds   = val_ds.map(tokenize_fn, batched=True)
for ds in (train_ds, val_ds):
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. Fine-tuning
training_args = TrainingArguments(
    output_dir=prune_model_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": (preds == p.label_ids).astype(float).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. Pruned + Fine-tuned 모델 저장
model.save_pretrained(prune_model_dir)
tokenizer.save_pretrained(prune_model_dir)

print(f"[완료] 구조적 Pruning 및 Fine-tuning 모델이 '{prune_model_dir}'에 저장되었습니다.")
