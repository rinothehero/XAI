import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

def extract_nonzero_rows(layer):
    weight = layer.weight.detach().cpu()
    row_norms = torch.norm(weight, p=1, dim=1)
    keep_rows = (row_norms != 0).nonzero(as_tuple=True)[0]
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
        # Intermediate.dense 축소
        old_inter = layer.intermediate.dense
        new_inter, inter_keep = shrink_linear_layer(old_inter)
        layer.intermediate.dense = new_inter

        # Output.dense 축소 (입력이 줄어든 것에 맞춰)
        old_output = layer.output.dense
        new_output = nn.Linear(len(inter_keep), old_output.out_features, bias=old_output.bias is not None)
        new_output.weight.data = old_output.weight.data[:, inter_keep].clone()
        if old_output.bias is not None:
            new_output.bias.data = old_output.bias.data.clone()
        layer.output.dense = new_output

        # (선택) Attention.dense 계층도 프루닝했으면 여기도 적용 가능

    return model

if __name__ == "__main__":
    model_path = "./pruned_bert_agnews_structured-20-6"
    save_path = "./shrunk_bert_agnews_structured-20-6"

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    shrunk_model = apply_structural_shrink(model)

    shrunk_model.save_pretrained(save_path)
    print(f"✅ 구조 축소 완료 및 저장: {save_path}")
