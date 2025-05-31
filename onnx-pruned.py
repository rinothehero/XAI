import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 모델 로드
model_path = "./pruned_bert_agnews_structured-20-6"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()  # ✅ 반드시 추론 모드로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Dummy input 생성 (ONNX는 고정된 입력 형태를 요구)
dummy_text = "This is a sample input sentence for ONNX export."
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 3. Export
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "pruned_bert.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"}
    },
    opset_version=16,
    do_constant_folding=True
)

print("✅ ONNX export 완료: pruned_bert.onnx")
