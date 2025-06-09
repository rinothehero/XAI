import pandas as pd

# 1. 두 CSV 파일 읽기
lime_q_8_df = pd.read_csv("../outputs/lime_explanation-8bit.csv")
lime_q_16_df = pd.read_csv("../outputs/lime_explanation-16bit.csv")
prun_quant_8_df = pd.read_csv("../outputs/lime_explanation-q_8_p-3.csv")
prun_quant_16_df = pd.read_csv("../outputs/lime_explanation-q16_p-3.csv")
prun_20_df = pd.read_csv("../outputs/lime_explanation-pruned-20-6.csv")
prun_30_df = pd.read_csv("../outputs/lime_explanation-pruned-40-1.csv")
sp_lime_df = pd.read_csv("../sp_lime_results/sp_lime_results.csv")


# 2. lime_explanation.csv에서 필요한 열만 선택하고 이름 바꾸기
lime_subset_8 = lime_q_8_df[["predicted_label", "gold_label_explanation", "lime_explanation"]].rename(columns={
    "predicted_label": "quant_8_label",
    "gold_label_explanation": "quant_8_gold_explanation",
    "lime_explanation": "quant_8_explanation"
})
lime_subset_16 = lime_q_16_df[["predicted_label", "gold_label_explanation", "lime_explanation"]].rename(columns={
    "predicted_label": "quant_16_label",
    "gold_label_explanation": "quant_16_gold_explanation",
    "lime_explanation": "quant_16_explanation"
})
prun_quant_8_subset = prun_quant_8_df[["predicted_label", "gold_label_explanation", "lime_explanation"]].rename(columns={
    "predicted_label": "pruned_quant_8_label",
    "gold_label_explanation": "pruned_quant_8_gold_explanation",
    "lime_explanation": "pruned_quant_8_explanation"
})

prun_quant_16_subset = prun_quant_16_df[["predicted_label", "gold_label_explanation", "lime_explanation"]].rename(columns={
    "predicted_label": "pruned_quant_16_label",
    "gold_label_explanation": "pruned_quant_16_gold_explanation",
    "lime_explanation": "pruned_quant_16_explanation"
})

prun_20_subset = prun_20_df[["predicted_label", "gold_label_explanation", "lime_explanation"]].rename(columns={
    "predicted_label": "pruned_20_label",
    "gold_label_explanation": "pruned_20_gold_explanation",
    "lime_explanation": "pruned_20_explanation"
})

prun_30_subset = prun_30_df[["predicted_label", "gold_label_explanation", "lime_explanation"]].rename(columns={
    "predicted_label": "pruned_30_label",
    "gold_label_explanation": "pruned_30_gold_explanation",
    "lime_explanation": "pruned_30_explanation"
})

# 3. 인덱스 기준으로 병합 (concat)
merged_df = pd.concat([sp_lime_df, lime_subset_8, lime_subset_16,prun_quant_8_subset, prun_quant_16_subset, prun_20_subset,prun_30_subset], axis=1)

# 4. 저장
merged_df.to_csv("./output_final.csv", index=False)