import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm

# 1. Load tokenizer and quantized model
model_path = "models/fine_tuned/bert_agnews"  # tokenizer 기준 경로
quantized_model_path = "models/quantized/bert_agnews_16bit.pt"  # 모델 경로

tokenizer = BertTokenizer.from_pretrained(model_path)
quantized_model = torch.load(quantized_model_path)
quantized_model.eval()

# (옵션) GPU 사용 가능 여부에 따라 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quantized_model.to(device)

# 2. Load AG News test set
dataset = load_dataset("ag_news", split="test")  # 7600개 샘플

# 3. Evaluation
correct = 0
total = 0
batch_size = 64

for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Quantized BERT"):
    batch = dataset[i:i + batch_size]
    texts = batch['text']
    labels = batch['label']

    # Tokenization
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor(labels).to(device)

    # Inference
    with torch.no_grad():
        outputs = quantized_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=1)

    correct += (preds == labels).sum().item()
    total += len(labels)

# 4. Print result
accuracy = correct / total
print(f"✅ Accuracy of Quantized BERT on AG News test set ({total} samples): {accuracy:.4f}")