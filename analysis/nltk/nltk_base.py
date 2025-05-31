import json
import nltk
from nltk import pos_tag
from collections import Counter


# NLTK 리소스 자동 다운로드 (처음 실행 시)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# JSON 파일 로드
with open('../sp_lime_results/sp_lime_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 품사 통계 Counter 초기화
pos_counter = Counter()

for sample in data:
    explanations = sample['pred_lime_explanation'][:5]
    words = [exp[0] for exp in explanations]
    tagged = pos_tag(words)  # [('word', 'POS'), ...]

    for word, tag in tagged:
        pos_counter[tag] += 1

# 결과 정리: 주요 품사 그룹핑
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

# 그룹핑하여 누적
grouped_counts = Counter()
for tag, count in pos_counter.items():
    group = pos_group_map.get(tag, f"Other({tag})")
    grouped_counts[group] += count

# 결과 출력
print("=== POS Category Counts (Top 5 LIME words x 100 samples) ===")
for category, count in grouped_counts.most_common():
    print(f"{category:<15}: {count}")

    # 핵심 품사 그룹 정의
core_pos = {"Noun", "Adjective"}

# 핵심 / 비핵심 품사 합계 계산
core_total = sum(count for cat, count in grouped_counts.items() if cat in core_pos)
other_total = sum(count for cat, count in grouped_counts.items() if cat not in core_pos)

print(f"\n핵심 품사 합계 (Noun, Proper Noun, Verb, Adjective): {core_total}")
print(f"기타 품사 합계: {other_total}")