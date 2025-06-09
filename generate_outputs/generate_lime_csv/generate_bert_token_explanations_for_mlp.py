#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import multiprocessing as mp
from datasets import load_dataset
from lime.lime_text import LimeTextExplainer

# (1) 워커 프로세스: GPU별로 실행
def lime_worker(gpu_id, texts, labels, out_path, model_path, class_names):
    import torch
    import torch.nn.functional as F
    from transformers import BertTokenizerFast, BertForSequenceClassification

    # (1-1) 각 프로세스에서 사용할 디바이스 지정
    device = torch.device(f"cuda:{gpu_id}")

    # (1-2) 토크나이저 및 모델 로드
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval().to(device)

    # (1-3) LIME Explainer
    explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=lambda txt: [
            txt[start:end]
            for start, end in tokenizer(
                txt,
                return_offsets_mapping=True,
                truncation=True,
                max_length=128
            )["offset_mapping"]
            if start < end
        ],
        bow=False
    )

    # (1-4) 예측 함수: 해당 GPU 사용
    def predict_proba(batch_texts):
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    # (1-5) LIME 설명 생성
    results = []
    for text, true_label in zip(texts, labels):
        offsets = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=128
        )["offset_mapping"]
        tokens = [text[start:end] for start, end in offsets if start < end]

        probs = predict_proba([text])[0]
        pred_label = int(probs.argmax())

        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=len(tokens),
            top_labels=len(class_names),
            num_samples=500
        )

        # 토큰별 기여도 매핑
        token_lime = {tok: [0.0] * len(class_names) for tok in tokens}
        for cls_idx, idx_weights in explanation.as_map().items():
            for idx, weight in idx_weights:
                token_lime[tokens[idx]][cls_idx] = weight

        results.append({
            "text": text,
            "true_label": int(true_label),
            "predicted_label": pred_label,
            "token_lime": token_lime,
            "probabilities": probs.tolist()
        })

    # (1-6) 결과 저장
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] done → {out_path}")


def main():
    # (2) 모델 경로 & 클래스명
    model_path = "../models/ft_BERT_agnews_full_dataset"
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    # (3) AG News에서 균형 샘플링
    per_class = 2500
    ds = load_dataset("ag_news", split="train")
    counts = {i: 0 for i in range(len(class_names))}
    buffer = []
    for ex in ds:
        lbl = ex["label"]
        if counts[lbl] < per_class:
            buffer.append(ex)
            counts[lbl] += 1
        if all(counts[i] == per_class for i in counts):
            break
    texts = [ex["text"] for ex in buffer]
    labels = [ex["label"] for ex in buffer]

    # (4) GPU 개수에 맞춰 데이터 분할
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("사용 가능한 GPU가 없습니다.")
    chunk_size = math.ceil(len(texts) / n_gpus)

    mp.set_start_method('spawn', force=True)
    procs = []
    for gpu_id in range(n_gpus):
        start = gpu_id * chunk_size
        end = min((gpu_id + 1) * chunk_size, len(texts))
        sub_texts = texts[start:end]
        sub_labels = labels[start:end]
        out_fn = f"../../outputs/intermediate_data/lime_gpu{gpu_id}.json"

        p = mp.Process(
            target=lime_worker,
            args=(gpu_id, sub_texts, sub_labels, out_fn, model_path, class_names)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # (5) 결과 병합
    merged = []
    for gpu_id in range(n_gpus):
        fn = f"lime_gpu{gpu_id}.json"
        with open(fn, "r", encoding="utf-8") as fin:
            merged.extend(json.load(fin))

    with open("../../outputs/intermediate_data/bert_token_explanations_for_mlp.json", "w", encoding="utf-8") as fout:
        json.dump(merged, fout, ensure_ascii=False, indent=2)
    print("✅ All GPUs done → lime_results_full.json")


if __name__ == "__main__":
    import torch   # torch.cuda.device_count() 위해
    main()