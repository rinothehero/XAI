import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Fine-tuned 모델 로드 (로컬 경로 또는 허깃허브)
model_name_or_path = "../prune-20/pruned_bert_agnews_structured-20-6"  # Fine-tuned 모델 경로
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=True)
#model = torch.load(model_name_or_path, map_location=torch.device('cpu'))
tokenizer = AutoTokenizer.from_pretrained('../models/ft_BERT_agnews_full_dataset', local_files_only=True)

# 2. CPU로 이동 및 eval 모드
model = model.half()
model.to("cpu")
model.eval()

# 4. 양자화 모델 저장
#quantized_model.save_pretrained("./quantized_bert_agnews_dynamic")
torch.save(model, "../quantized-bert/quantized16_pruned_dynamic-3.pt")
#tokenizer.save_pretrained("./quantized_pruned_dynamic")

print("Dynamic Quantization 완료 및 저장됨: ./quantized_pruned_dynamic")
