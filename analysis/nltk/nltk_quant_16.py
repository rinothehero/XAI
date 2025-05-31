import pandas as pd
import nltk
from nltk import pos_tag
from collections import Counter

# NLTK 리소스 다운로드 (최초 1회만 실행됨)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# CSV 파일 로드
df = pd.read_csv("../outputs/lime_explanation-16bit.csv")

# 품사 그룹핑 맵
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

# 품사 카운터 초기화
grouped_counts = Counter()

# 각 행에서 lime_explanation 파싱
for explanation in df["lime_explanation"].dropna():
    # 문자열에서 상위 5개 단어 추출 (형식: word(score); ...)
    parts = explanation.split(';')[:5]
    words = [p.strip().split('(')[0] for p in parts if '(' in p]

    tagged = pos_tag(words)
    for word, tag in tagged:
        group = pos_group_map.get(tag, f"Other({tag})")
        grouped_counts[group] += 1

# 결과 출력
print("=== POS Category Counts (Top 5 LIME words x 100 samples) ===")
for category, count in grouped_counts.most_common():
    print(f"{category:<15}: {count}")

# 핵심 품사 그룹 정의
core_pos = {"Noun", "Adjective","Proper Noun"}

# 핵심 / 비핵심 품사 합계 계산
core_total = sum(count for cat, count in grouped_counts.items() if cat in core_pos)
other_total = sum(count for cat, count in grouped_counts.items() if cat not in core_pos)

print(f"\n핵심 품사 합계 (Noun, Proper Noun, Verb, Adjective): {core_total}")
print(f"기타 품사 합계: {other_total}")
