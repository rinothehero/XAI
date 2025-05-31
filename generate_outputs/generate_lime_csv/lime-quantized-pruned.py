from transformers import BertTokenizer
from lime.lime_text import LimeTextExplainer
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# 1. Load tokenizer
model_path = "../models/ft_BERT_agnews_full_dataset"  # 학습 시 사용한 토크나이저 경로
tokenizer = BertTokenizer.from_pretrained(model_path)

# 2. Load quantized model
quantized_model = torch.load("../quantized-bert/quantized_pruned_dynamic-3.pt")
quantized_model.eval()

# 3. Define class names
class_names = ["World", "Sports", "Business", "Sci/Tech"]

# 4. Define LIME-compatible prediction function
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = quantized_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    return probs.numpy()

# 5. Instantiate LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

# 6. Input text to explain
input_file = "../sp_lime_results/sp_lime_results.csv"
df = pd.read_csv(input_file)
#text_df = df[["text"],["gold_label"],["predicted_label"]]
output_file = "lime_text_only.csv"
df.to_csv(output_file, index=False)

text_df = pd.read_csv(output_file)
results = []
wrong = []
diff_from_bert = []

start_time = time.time()
# 7. LIME explanation for each text (예시: 처음 1개 또는 N개만)
for i, row in tqdm(text_df.iterrows(), total=len(text_df), desc="Generating LIME Explanations"):  # 숫자 바꿔서 여러 개 해석 가능
    text = row["text"]
    gold_label = row["gold_label"]
    predicted_label = row["predicted_label"]

    #print(f"\n🟡 Text #{i+1}: {text[:80]}...")
    explanation = explainer.explain_instance(text, predict_proba, num_features=100, top_labels=4)
    label = explanation.top_labels[0]
    explanation_list = explanation.as_list(label=label)

    # format: token1(score1); token2(score2); ...
    explanation_str = "; ".join([f"{word}({score:.4f})" for word, score in explanation_list])
    gold_explanation = ""
    predicted_explanation = ""

    if gold_label != label:
        wrong.append(i)
        gold_explanation_list = explanation.as_list(label=gold_label)
        gold_explanation = "; ".join([f"{word}({score:.4f})" for word, score in gold_explanation_list])
    
    if predicted_label != label:
        diff_from_bert.append(i)
        predicted_explanation_list = explanation.as_list(label=predicted_label)
        predicted_explanation = "; ".join([f"{word}({score:.4f})" for word, score in predicted_explanation_list])
    
    results.append({
        "text": text,
        "predicted_label": label,
        "gold_label": gold_label,
        "BERT predicted_label": predicted_label,
        "lime_explanation": explanation_str,
        "gold_label_explanation": gold_explanation,
        "predicted_label_explanation": predicted_explanation,
    })
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
# 8. Show explanation
#for label, importance in explanation.as_list(label=explanation.top_labels[0]):
#    print(f"{label:20s}: {importance:.4f}")
output_df = pd.DataFrame(results)
output_df.to_csv("lime_explanation-pruned-1.csv", index=False)

print("wrong: ", wrong)
print("diff_from_bert: ", diff_from_bert)
# 또는 출력만: print(explanation.as_list())
