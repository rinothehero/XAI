import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lime.lime_text import LimeTextExplainer
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def word_to_idx(word: str, dim: int) -> int:
    """
    Reproducible hash-based mapping of a word to an index in [0, dim).
    """
    h = hashlib.md5(word.encode("utf-8")).hexdigest()
    return int(h, 16) % dim


def load_model_and_tokenizer(model_dir: str, device: str = "cuda"):
    """
    Load model and tokenizer, move model to specified device.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    #model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model = torch.load(model_dir, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained("./models/ft_BERT_agnews_full_dataset", local_files_only=True)
    model = model.to(device).eval()
    return model, tokenizer


def sample_texts(dataset_name: str, split: str, n_per_class: int, seed: int = 42) -> (List[str], List[int]):
    """
    Stratified sampling of `n_per_class` texts per class from the dataset split.
    Returns texts list and labels list.
    """
    dataset = load_dataset(dataset_name)
    data = dataset[split]
    classwise = defaultdict(list)
    for item in data:
        classwise[item["label"]].append(item["text"])
    random.seed(seed)
    texts, labels = [], []
    for lbl, items in classwise.items():
        random.shuffle(items)
        selected = items[:n_per_class]
        texts.extend(selected)
        labels.extend([lbl] * len(selected))
    return texts, labels


def predict_proba(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 32
) -> np.ndarray:
    """
    Predict probability on given device in batches.
    """
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(all_probs)


def compute_lime_matrix(
    texts: List[str],
    preds: List[int],
    golds: List[int],
    model: torch.nn.Module,
    tokenizer,
    num_features: int,
    num_samples: int,
    hash_dim: int,
    seed: int,
    device: str = "cpu"
) -> (csr_matrix, List):
    """
    Generate LIME explanations for each sample's predicted class and build a sparse feature matrix.
    """
    model.to(device)
    explainer = LimeTextExplainer(class_names=None, random_state=seed)
    n = len(texts)
    X_lil = lil_matrix((n, hash_dim), dtype=np.float32)
    explanations = []
    for i, (text, pred, gold) in enumerate(tqdm(zip(texts, preds, golds), total=n, desc="LIME explanations")):
        labels_to_explain = list({pred, gold})
        exp = explainer.explain_instance(
            text,
            lambda ts: predict_proba(ts, model, tokenizer, device),
            num_features=num_features,
            num_samples=num_samples,
            labels=labels_to_explain
        )
        for word, weight in exp.as_list(label=pred):
            idx = word_to_idx(word, hash_dim)
            X_lil[i, idx] = weight
        explanations.append(exp)
    return X_lil.tocsr(), explanations


def select_sp_lime(sim_matrix: np.ndarray, B: int) -> List[int]:
    """
    Greedy submodular pick on precomputed similarity matrix to maximize explanation diversity.
    """
    n = sim_matrix.shape[0]
    selected = []
    summed_sim = np.zeros(n)
    for _ in range(B):
        sims = sim_matrix[:, selected].sum(axis=1) if selected else np.zeros(n)
        scores = sims + summed_sim
        pick = int(np.argmin(scores))
        selected.append(pick)
        summed_sim += sim_matrix[:, pick]
    return selected


def save_results(picked_info: List[dict], output_prefix: str):
    """
    Save picked samples info to JSON and CSV, ensuring output directory exists.
    """
    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)
    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(picked_info, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame([{  
        "text": rec["text"],
        "gold_label": rec["gold_label"],
        "predicted_label": rec["predicted_label"],
        "pred_prob"     : rec["pred_prob"],
        "pred_lime": "; ".join(f"{w}({wt})" for w, wt in rec["pred_lime_explanation"]),
        "gold_lime": ("; ".join(f"{w}({wt})" for w, wt in rec["gold_lime_explanation"])
                    if rec["gold_lime_explanation"] else "")
    } for rec in picked_info])
    df.to_csv(f"{output_prefix}.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="SP-LIME pipeline with sparse matrix and precomputed similarity for predicted classes.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="ag_news")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_per_class", type=int, default=100)
    parser.add_argument("--num_features", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=500, help="LIME perturbation samples")
    parser.add_argument("--hash_dim", type=int, default=5000)
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lime_device", type=str, default="cpu", choices=["cpu","cuda"], help="Device for LIME explain_instance")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_prefix", type=str, default="../../outputs/sp_lime_sets/torch_loaded_model_sp_lime")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #base_device = "cuda:1" if torch.cuda.is_available() else "cpu"
    base_device = "cpu"  # CPU로 강제 설정
    model, tokenizer = load_model_and_tokenizer(args.model_dir, base_device)

    texts, labels = sample_texts(args.dataset, args.split, args.n_per_class, seed=args.seed)
    probs = predict_proba(texts, model, tokenizer, base_device, batch_size=args.batch_size)
    preds = np.argmax(probs, axis=1).tolist()

    X, exps = compute_lime_matrix(
        texts, preds, labels,           # ← golds=labels 추가
        model, tokenizer,
        num_features=args.num_features,
        num_samples=args.num_samples,
        hash_dim=args.hash_dim,
        seed=args.seed,
        device=args.lime_device
    )

    sim = cosine_similarity(X, X)
    picked = select_sp_lime(sim, args.B)

    picked_info = []
    for rank, idx in enumerate(picked, 1):
        text = texts[idx]
        gold = labels[idx]
        pred = preds[idx]
        pred_prob = float(probs[idx, pred])          # ★ 추가
         # 예측 클래스 설명
        pred_exp = [(w, round(float(wt), 4))
                    for w, wt in exps[idx].as_list(label=pred)]

        # 정답 클래스 설명 (동일하면 None)
        gold_exp = None
        if gold != pred:
            gold_exp = [(w, round(float(wt), 4))
                        for w, wt in exps[idx].as_list(label=gold)]
        picked_info.append({
            "index": rank,
            "text": text,
            "gold_label": gold,
            "predicted_label": pred,
            "pred_prob": round(pred_prob, 4),
            "pred_lime_explanation": pred_exp,   # 📄 새 컬럼
            "gold_lime_explanation": gold_exp    # 📄 새 컬럼 (또는 None)
        })

    save_results(picked_info, args.output_prefix)

if __name__ == "__main__":
    main()
# 14m 57s