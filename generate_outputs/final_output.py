import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from pathlib import Path

# 폴더 경로 설정
input_dir = Path("../gold_quant_merge")

# 유틸 함수
def safe_cosine(v1, v2):
    if np.all(v1 == 0) or np.all(v2 == 0):
        return np.nan
    return 1 - cosine(v1, v2)

# 결과를 담을 리스트
results = []

# 100개 샘플 반복
for idx in range(100):
    file_path = input_dir / f"sample_{idx:03d}.csv"
    
    try:
        df = pd.read_csv(file_path)
        
        # 벡터 추출
        base = df["base_lime_value"].to_numpy()
        q8 = df["quant_8_lime_value"].to_numpy()
        q16 = df["quant_16_lime_value"].to_numpy()
        prunedquant8 = df["pruned_quant_8_lime_value"].to_numpy()
        prunedquant16 = df["pruned_quant_16_lime_value"].to_numpy()
        pruned20 = df["pruned_20_lime_value"].to_numpy()
        pruned30 = df["pruned_30_lime_value"].to_numpy()
        
        # 코사인 유사도
        cos_q8 = safe_cosine(base, q8)
        cos_q16 = safe_cosine(base, q16)
        cos_prunedquant8 = safe_cosine(base, prunedquant8)
        cos_prunedquant16 = safe_cosine(base, prunedquant16)
        cos_pruned20 = safe_cosine(base, pruned20)
        cos_pruned30 = safe_cosine(base, pruned30)
        
        # 스피어만 상관계수
        spearman_q8, _    = spearmanr(base, q8)
        spearman_q16, _   = spearmanr(base, q16)
        spearman_prunedquant8, _ = spearmanr(base, prunedquant8)
        spearman_prunedquant16, _ = spearmanr(base, prunedquant16)
        spearman_pruned20, _ = spearmanr(base, pruned20)
        spearman_pruned30, _ = spearmanr(base, pruned30)
        
        # 리스트에 추가
        results.append({
            "sample": idx + 1,
            "cosine_q8": cos_q8,
            "spearman_q8": spearman_q8,
            "cosine_q16": cos_q16,
            "spearman_q16": spearman_q16,
            "cosine_prunedquant8": cos_prunedquant8,
            "spearman_prunedquant8": spearman_prunedquant8,
            "cosine_prunedquant16": cos_prunedquant16,
            "spearman_prunedquant16": spearman_prunedquant16,
            "cosine_pruned20": cos_pruned20,
            "spearman_pruned20": spearman_pruned20,
            "cosine_pruned30": cos_pruned30,
            "spearman_pruned30": spearman_pruned30
        })
        
        print(f"[{idx+1:3d}] Cosine (q8): {cos_q8:.4f} | Spearman (q8): {spearman_q8:.4f} | "
              f"Cosine (q16): {cos_q16:.4f} | Spearman (q16): {spearman_q16:.4f}"
              f" | Cosine (pruned quant): {cos_prunedquant8:.4f} | Spearman (pruned quant): {spearman_prunedquant8:.4f}"
              f" | Cosine (pruned quant 16): {cos_prunedquant16:.4f} | Spearman (pruned quant 16): {spearman_prunedquant16:.4f}"
              f" | Cosine (pruned 20): {cos_pruned20:.4f} | Spearman (pruned 20): {spearman_pruned20:.4f}"
              f" | Cosine (pruned 30): {cos_pruned30:.4f} | Spearman (pruned 30): {spearman_pruned30:.4f}")    
    except Exception as e:
        print(f"[{idx+1:3d}] Error reading {file_path.name}: {e}")

# DataFrame으로 변환 후 CSV로 저장
output_df = pd.DataFrame(results)
output_df.to_csv("true_final_output.csv", index=False)
print("▶ 최종 결과가 true_final_output.csv로 저장되었습니다.")
