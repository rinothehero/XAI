import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Fine-tuned 모델 로드 (로컬 경로 또는 허깃허브)
model_name_or_path = "./models/ft_BERT_agnews_full_dataset"  # Fine-tuned 모델 경로
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)

# 2. CPU로 이동 및 eval 모드
model.to("cpu")
model.eval()

# 3. Dynamic Quantization 적용 (Linear 레이어만 양자화)
quantized_model = torch.quantization.quantize_dynamic(
    model,                     # 양자화할 원본 모델
    {torch.nn.Linear},         # 양자화 대상 모듈 클래스
    dtype=torch.qint8          # 양자화 비트 폭 (qint8 또는 quint8)
)

# 4. 양자화 모델 저장
#quantized_model.save_pretrained("./quantized_bert_agnews_dynamic")
torch.save(quantized_model, "./quantized_bert_agnews_dynamic.pt") # 173.11MB
tokenizer.save_pretrained("./quantized_bert_agnews_dynamic")

print("Dynamic Quantization 완료 및 저장됨: ./quantized_bert_agnews_dynamic")