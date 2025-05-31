import torch
from transformers import AutoTokenizer

# 구조 축소 모델 로드
model = torch.load("../../shrunk_bert_state-prune-20.pt", map_location="cpu")
model.eval()

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("../../models/ft_BERT_agnews_full_dataset")

# Dummy input
text = "This is a test sentence for ONNX export."
inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

# ONNX export
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "shrunk_prune20.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=16
)

print("✅ ONNX export 완료 → shrunk_prune20.onnx")
