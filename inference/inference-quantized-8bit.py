from transformers import BertTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
import time
from tqdm import tqdm

# 1. Load tokenizer and quantized model
model_path = "../models/ft_BERT_agnews_full_dataset"
tokenizer = BertTokenizer.from_pretrained(model_path)

quantized_model = torch.load("../quantized-bert/quantized_bert_agnews_dynamic.pt")
quantized_model.eval()

# 2. Load input data
input_file = "../sp_lime_results/sp_lime_results.csv"
df = pd.read_csv(input_file)

# 3. Set up counters and timing
correct = 0
total = 0
batch_size = 64
start_time = time.time()

# 4. Run inference
for i in range(100):
    for i in tqdm(range(0, len(df), batch_size), desc="Running Inference"):
        batch = df.iloc[i:i + batch_size]
        text = batch["text"].tolist()
        gold_label = batch["gold_label"].tolist()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = quantized_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).tolist()

        correct += sum([int(p == g) for p, g in zip(pred, gold_label)])
        total += len(gold_label)

end_time = time.time()

# 5. Calculate metrics
accuracy = correct / total
total_time = end_time - start_time
avg_time = total_time / total

# 6. Save summary to CSV
summary = {
    "total_instances": [total],
    "correct_predictions": [correct],
    "accuracy": [accuracy],
    "total_time_sec": [total_time],
    "avg_time_per_instance_sec": [avg_time]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv("./metrics/inference_summary-8bit.csv", index=False)

print("✅ Inference completed. Summary saved to 'inference_summary.csv'")
