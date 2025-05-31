import pandas as pd
import re
from pathlib import Path

# CSV 파일 로드
df = pd.read_csv("output_final.csv")  # 실제 파일 이름으로 변경

def parse_explanation(text):
    """
    "word(0.123); other(-0.456)" → dict(word: 0.123, other: -0.456)
    """
    if pd.isna(text):
        return {}
    pattern = r'([^;()]+)\((-?\d+\.\d+)\)'
    return {match[0].strip(): float(match[1]) for match in re.findall(pattern, text)}

# 결과 저장 디렉토리 생성
output_dir = Path("../gold_quant_merge")
output_dir.mkdir(exist_ok=True)

for idx, row in df.iterrows():
    # 1. base LIME 결정
    if row["gold_label"] == row["predicted_label"]:
        base_lime_dict = parse_explanation(row.get("pred_lime"))
    else:
        base_lime_dict = parse_explanation(row.get("gold_lime"))

    # 2. quant LIME 결정
    if row["quant_8_label"] == row["gold_label"]:
        quant_8_lime_dict = parse_explanation(row.get("quant_8_explanation"))
    else:
        quant_8_lime_dict = parse_explanation(row.get("quant_8_gold_explanation"))
    
    if row["quant_16_label"] == row["gold_label"]:
        quant_16_lime_dict = parse_explanation(row.get("quant_16_explanation"))
    else:   
        quant_16_lime_dict = parse_explanation(row.get("quant_16_gold_explanation"))

    if row["pruned_quant_8_label"] == row["gold_label"]:
        pruned_quant_8_lime_dict = parse_explanation(row.get("pruned_quant_8_explanation"))
    else:   
        pruned_quant_8_lime_dict = parse_explanation(row.get("pruned_quant_8_gold_explanation"))
        
    if row["pruned_quant_16_label"] == row["gold_label"]:
        pruned_quant_16_lime_dict = parse_explanation(row.get("pruned_quant_16_explanation"))
    else:   
        pruned_quant_16_lime_dict = parse_explanation(row.get("pruned_quant_16_gold_explanation"))

    if row["pruned_20_label"] == row["gold_label"]:
        pruned_20_lime_dict = parse_explanation(row.get("pruned_20_explanation"))
    else:   
        pruned_20_lime_dict = parse_explanation(row.get("pruned_20_gold_explanation"))

    if row["pruned_30_label"] == row["gold_label"]:
        pruned_30_lime_dict = parse_explanation(row.get("pruned_30_explanation"))
    else:   
        pruned_30_lime_dict = parse_explanation(row.get("pruned_30_gold_explanation"))

    # 3. 공통 feature만 병합
    merged = []
    for feat in base_lime_dict:
        if feat in quant_8_lime_dict and feat in quant_16_lime_dict and feat in pruned_quant_8_lime_dict and feat in pruned_quant_16_lime_dict and feat in pruned_20_lime_dict and feat in pruned_30_lime_dict:
            merged.append({
                "feature": feat,
                "base_lime_value": base_lime_dict[feat],
                "quant_8_lime_value": quant_8_lime_dict[feat],
                "quant_16_lime_value": quant_16_lime_dict[feat],
                "pruned_quant_8_lime_value": pruned_quant_8_lime_dict[feat],
                "pruned_quant_16_lime_value": pruned_quant_16_lime_dict[feat],
                "pruned_20_lime_value": pruned_20_lime_dict[feat],
                "pruned_30_lime_value": pruned_30_lime_dict[feat]
            })

    # 4. 저장
    merged_df = pd.DataFrame(merged)
    merged_df.to_csv(output_dir / f"sample_{idx:03d}.csv", index=False)