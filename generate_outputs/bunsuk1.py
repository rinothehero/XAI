import pandas as pd

# 1. 파일 불러오기
df = pd.read_csv("output_final.csv")
# 2. 원하는 열 순서로 재정렬
desired_order = [
    "text",
    "pred_prob",
    "gold_label",
    "predicted_label",
    "quant_8_label",
    "quant_16_label",
    "pruned_quant_8_label",
    "pruned_quant_16_label",
    "pruned_20_label",
    "pruned_30_label",
    "gold_lime",
    "pred_lime",
    "quant_8_gold_explanation",
    "quant_8_explanation",
    "quant_16_gold_explanation",
    "quant_16_explanation",
    "pruned_quant_8_gold_explanation",
    "pruned_quant_8_explanation",
    "pruned_quant_16_gold_explanation",
    "pruned_quant_16_explanation",
    "pruned_20_gold_explanation",
    "pruned_20_explanation",
    "pruned_30_gold_explanation",
    "pruned_30_explanation",
]
df = df[desired_order]

# 3. 다시 저장
df.to_csv("output_final.csv", index=False)
