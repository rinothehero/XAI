from .shrinking_utils import apply_structural_shrink
# shrink_and_export.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 🔁 MAIN: shrink + export
model_path = "models/pruned/bert_agnews_20pct_v1" # Placeholder, ensure this model exists or adjust path
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

model = apply_structural_shrink(model)

# 🟩 dummy input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_text = "This is a test sentence."
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 🟩 ONNX export
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "models/onnx/shrunk_bert-prune-20.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"}
    },
    opset_version=16,
    do_constant_folding=True,
    verbose=True
)

print("✅ ONNX export 완료 → models/onnx/shrunk_bert-prune-20.onnx")
