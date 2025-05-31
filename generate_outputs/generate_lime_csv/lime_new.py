import torch
import numpy as np
import json
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

# 1. Load tokenizer and model
model_path = "../models/ft_BERT_agnews_full_dataset"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval().cuda()

# 2. Load and sample AG News
dataset = load_dataset("ag_news", split="train")
samples_per_class = 3
class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
selected_samples = []

for ex in dataset:
    label = ex["label"]
    if class_counts[label] < samples_per_class:
        selected_samples.append(ex)
        class_counts[label] += 1
    if all(c == samples_per_class for c in class_counts.values()):
        break

texts = [ex["text"] for ex in selected_samples]
true_labels = [ex["label"] for ex in selected_samples]

# 3. Define prediction function
class_names = ["World", "Sports", "Business", "Sci/Tech"]

def predict_proba(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs).logits
        probs = F.softmax(outputs, dim=1).cpu().numpy()
    return probs

# 4. LIME Explainer
explainer = LimeTextExplainer(class_names=class_names)

# 5. For each sample, get class-wise LIME scores
results = []
for i, text in enumerate(tqdm(texts)):
    true_label = int(true_labels[i])
    probs = predict_proba([text])[0]
    pred_label = int(np.argmax(probs))
    
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=20,
        top_labels=len(class_names)
    )

    # Build token_lime[token] → List[float]: 각 클래스별 중요도 벡터
    token_lime = {}
    for class_idx in range(len(class_names)):
        for word, score in explanation.as_list(label=class_idx):
            if word not in token_lime:
                token_lime[word] = [0.0] * len(class_names)
            token_lime[word][class_idx] = score

    result = {
        "text": text,
        "true_label": true_label,
        "predicted_label": pred_label,
        "token_lime": token_lime,
        "probabilities": probs.tolist()
    }
    results.append(result)

# 6. Save
with open("lime_results_full.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Results saved to lime_results_full.json")