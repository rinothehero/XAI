# shrink_and_export.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def extract_nonzero_rows(layer):
    weight = layer.weight.detach().cpu()
    row_norms = torch.norm(weight, p=1, dim=1)
    keep_rows = (row_norms != 0).nonzero(as_tuple=True)[0]
    if len(keep_rows) == 0:
        print("⚠️ 모든 row가 pruning됨. 최소 1개 유지합니다.")
        keep_rows = torch.tensor([0])
    return keep_rows

def shrink_linear_layer(old_layer):
    keep_rows = extract_nonzero_rows(old_layer)
    in_features = old_layer.in_features
    out_features = len(keep_rows)

    new_layer = nn.Linear(in_features, out_features, bias=old_layer.bias is not None)
    new_layer.weight.data = old_layer.weight.data[keep_rows].clone()
    if old_layer.bias is not None:
        new_layer.bias.data = old_layer.bias.data[keep_rows].clone()
    return new_layer, keep_rows

def apply_structural_shrink(model):
    for i, layer in enumerate(model.bert.encoder.layer):
        old_inter = layer.intermediate.dense
        new_inter, keep_rows = shrink_linear_layer(old_inter)
        layer.intermediate.dense = new_inter

        old_output = layer.output.dense
        new_output = nn.Linear(len(keep_rows), old_output.out_features, bias=old_output.bias is not None)
        new_output.weight.data = old_output.weight.data[:, keep_rows].clone()
        if old_output.bias is not None:
            new_output.bias.data = old_output.bias.data.clone()
        layer.output.dense = new_output
    return model

# 🔁 MAIN: shrink + export
model_path = "./pruned_bert_agnews_structured-30-6"
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
    "shrunk_bert-prune-30.onnx",
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

print("✅ ONNX export 완료 → shrunk_bert.onnx")
