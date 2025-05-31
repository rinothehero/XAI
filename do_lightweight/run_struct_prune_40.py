import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# 설정
model_path = "../models/ft_BERT_agnews_full_dataset"
save_path  = "../pruned_bert_agnews_structured-40-1"
os.makedirs(save_path, exist_ok=True)
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 1
LR = 1e-6

# 모델·토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prune 대상 수집
to_prune = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        to_prune.append((name, module, "weight"))

# Structured pruning 적용
for _, module, weight_name in to_prune:
    prune.ln_structured(module, name=weight_name, amount=0.4, n=1, dim=0)

# pruning 확정 (reparameterization 제거)
for _, module, weight_name in to_prune:
    prune.remove(module, weight_name)

# Zero row 위치 저장 (학습 중 gradient masking용)
zero_row_masks = {}
for name, module, _ in to_prune:
    weight = module.weight.detach().cpu()
    row_norms = torch.norm(weight, p=1, dim=1)
    zero_rows = (row_norms == 0).to(torch.bool)
    zero_row_masks[name] = zero_rows.to(DEVICE)

# AG News 데이터셋 로드 및 전처리
dataset = load_dataset("ag_news")
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds   = split["test"]

def tokenize_fn(ex):
    return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds   = val_ds.map(tokenize_fn, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

MAX_STEPS = int(len(train_loader) * 0.5)  # 0.5 epoch
step_count = 0

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ✅ Custom training loop with gradient masking
model.train()
for epoch in range(EPOCHS):
    print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
    for batch in tqdm(train_loader):
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(DEVICE)

        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()

        # ✅ Pruned row의 gradient = 0 (gradient masking)
        for name, module, _ in to_prune:
            if module.weight.grad is not None:
                mask = zero_row_masks[name].unsqueeze(1)  # (rows, 1)
                module.weight.grad[mask.expand_as(module.weight)] = 0

        optimizer.step()
        step_count += 1
        if step_count >= MAX_STEPS:
            print(f"[Early Stop] {step_count} steps reached (0.5 epoch)")
            break  # 내부 for-loop 종료
    if step_count >= MAX_STEPS:
        break  # 외부 epoch-loop도 종료

# 모델 저장
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\n[완료] pruning + gradient masking 학습 모델 저장됨: {save_path}")
