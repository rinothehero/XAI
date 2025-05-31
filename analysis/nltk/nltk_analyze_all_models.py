import pandas as pd
import json
import nltk
from nltk import pos_tag
from collections import Counter

# 사용자 입력
K = int(input("몇 개의 단어를 분석하시겠습니까? (기본값: 5): ") or 5)

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 품사 그룹 맵
pos_group_map = {
    "NN": "Noun", "NNS": "Noun", "NNP": "Proper Noun", "NNPS": "Proper Noun",
    "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", "VBZ": "Verb",
    "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective",
    "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb",
    "IN": "Preposition", "PRP": "Pronoun", "PRP$": "Pronoun", "WP": "Pronoun", "WP$": "Pronoun",
    "DT": "Determiner", "CC": "Conjunction", "UH": "Interjection",
    "CD": "Number", "TO": "to", "EX": "Existential", "MD": "Modal",
    "POS": "Possessive", "FW": "Foreign", "SYM": "Symbol", "LS": "List marker",
    "WDT": "Wh-determiner", "WRB": "Wh-adverb"
}

# 핵심 품사 및 맥락적 품사 정의
core_pos = {"Noun", "Adjective"}
contextual_pos = {"Preposition", "Conjunction", "to", "Modal"}

# 입력 파일 목록
csv_files = [
    "../outputs/lime_explanation-16bit.csv",
    "../outputs/lime_explanation-8bit.csv",
    "../outputs/lime_explanation-pruned-20-6.csv",
    "../outputs/lime_explanation-pruned-30-6.csv",
    "../outputs/lime_explanation-pruned-40-1.csv",
    "../outputs/lime_explanation-q_8_p-3.csv",
    "../outputs/lime_explanation-q16_p-3.csv",
]

results = []

def process_counter(pos_counter, source_name):
    total = sum(pos_counter.values())
    core_total = sum(count for pos, count in pos_counter.items() if pos in core_pos)
    context_total = sum(count for pos, count in pos_counter.items() if pos in contextual_pos)
    other_total = total - core_total - context_total
    return {
        "Source": source_name,
        "Core Ratio (%)": round(core_total / total * 100, 2) if total else 0,
        "Contextual Ratio (%)": round(context_total / total * 100, 2) if total else 0,
        "Other Ratio (%)": round(other_total / total * 100, 2) if total else 0
    }

# JSON 파일 처리
with open('../sp_lime_results/sp_lime_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pos_counter = Counter()
for sample in data:
    explanations = sample['pred_lime_explanation'][:K]
    words = [exp[0] for exp in explanations]
    tagged = pos_tag(words)
    for word, tag in tagged:
        group = pos_group_map.get(tag, f"Other({tag})")
        pos_counter[group] += 1

results.append(process_counter(pos_counter, "sp_lime_results.json"))

# CSV 파일들 처리
for file_path in csv_files:
    df = pd.read_csv(file_path)
    pos_counter = Counter()
    for explanation in df["lime_explanation"].dropna():
        parts = explanation.split(';')[:K]
        words = [p.strip().split('(')[0] for p in parts if '(' in p]
        tagged = pos_tag(words)
        for word, tag in tagged:
            group = pos_group_map.get(tag, f"Other({tag})")
            pos_counter[group] += 1
    results.append(process_counter(pos_counter, file_path.split("/")[-1]))

# 결과 출력
df_result = pd.DataFrame(results)
df_result.set_index("Source", inplace=True)
print("\n=== 핵심 / 맥락 / 기타 품사 비율 요약 ===")
print(df_result)
