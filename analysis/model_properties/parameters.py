import torch

# 모델 로드 (구조 + 가중치 포함된 경우)
model = torch.load("generate_lime_csv/EXP/a/models/mlp_xai_agnews_manualIG_0.5.pt", map_location="cpu")
model.eval()

# 전체 파라미터 수
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"📦 총 파라미터 수: {total_params:,}")
print(f"🧠 학습 가능한 파라미터 수: {trainable_params:,}")
