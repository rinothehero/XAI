import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Fine-tuned 모델 로드 (로컬 경로 또는 허깃허브)
model_name_or_path = "../models/ft_BERT_agnews_full_dataset"  # Fine-tuned 모델 경로
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)

# 2. CPU로 이동 및 eval 모드
model = model.half().to("cuda")
model.eval()

# 3. 양자화 모델 저장
#quantized_model.save_pretrained("./quantized_bert_agnews_dynamic")
torch.save(model, "../quantized-bert/quantized_bert_agnews_16bit.pt") # 173.11MB

print("Dynamic Quantization 완료 및 저장됨: ./quantized_bert_agnews_16-bit.pt")