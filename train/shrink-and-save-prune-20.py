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
        # 축소된 intermediate.dense 생성
        old_inter = layer.intermediate.dense
        new_inter, keep_rows = shrink_linear_layer(old_inter)
        layer.intermediate.dense = new_inter

        # output.dense: 축소된 intermediate에 맞게 입력 크기 수정
        old_output = layer.output.dense
        new_output = nn.Linear(len(keep_rows), old_output.out_features, bias=old_output.bias is not None)
        new_output.weight.data = old_output.weight.data[:, keep_rows].clone()
        if old_output.bias is not None:
            new_output.bias.data = old_output.bias.data.clone()
        layer.output.dense = new_output
    return model

if __name__ == "__main__":
    # 1. 원본 프루닝된 모델 불러오기
    model_path = "./pruned_bert_agnews_structured-20-6"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # 2. 구조 축소 적용
    model = apply_structural_shrink(model)

    # 3. state_dict 저장
    torch.save(model, "./shrunk_bert_state.pt")
    print("✅ 구조 축소 완료 및 .pt 저장: ./shrunk_bert_state.pt")
