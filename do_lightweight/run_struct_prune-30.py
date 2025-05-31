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

# 0. 설정
orig_model_dir = "../models/ft_BERT_agnews_full_dataset"  # Fine-tuned 모델 경로
prune_model_dir = "../pruned_bert_agnews_structured-30"
os.makedirs(prune_model_dir, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 모델·토크나이저 로드
model     = AutoModelForSequenceClassification.from_pretrained(orig_model_dir).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(orig_model_dir)

# 2. 구조적 Pruning 대상 수집 (모든 Linear 레이어)
to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        to_prune.append((module, 'weight'))

# 3. Structured Pruning 적용
#    - dim=0 으로 “output features (rows)” 단위로 L1-norm 하위 20% 제거
for module, weight_name in to_prune:
    prune.ln_structured(
        module,
        name=weight_name,
        amount=0.3,   # 제거 비율: 30%
        n=1,          # L1-norm
        dim=0         # 행(row) 단위로 prune
    )

# 4. Pruning 마스크를 weight에 고정 (Reparameterization 제거)
#for module, weight_name in to_prune:
#    prune.remove(module, weight_name)

# 5. 데이터셋 로드 및 전처리 (AG News 예시)
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

# 6. 재미세조정(Fine-tune)
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

for module, weight_name in to_prune:
    prune.remove(module, weight_name)

# 7. Pruned + Fine-tuned 모델 저장
model.save_pretrained(prune_model_dir)
tokenizer.save_pretrained(prune_model_dir)

print(f"[완료] 구조적 Pruning 및 Fine-tuning 모델이 '{prune_model_dir}'에 저장되었습니다.")