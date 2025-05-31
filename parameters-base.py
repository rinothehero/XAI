from transformers import AutoModelForSequenceClassification

# 체크포인트 경로
model_path = "./models/ft_BERT_agnews_full_dataset/checkpoint-5064"

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"📦 총 파라미터 수: {total_params:,}")
print(f"🧠 학습 가능한 파라미터 수: {trainable_params:,}")
