import onnxruntime as ort
import numpy as np
import time
from transformers import AutoTokenizer
from tqdm import tqdm

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("../../models/ft_BERT_agnews_full_dataset")

# 2. Load ONNX model using GPU (CUDAExecutionProvider)
onnx_model_path = "shrunk_prune20.onnx"
sess = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])
print(" 현재 사용 중인 Execution Provider:", sess.get_providers())

# 3. Prepare dummy input
text = "This is a test sentence for measuring ONNX inference speed."
inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)

input_ids = inputs["input_ids"].astype(np.int64)
attention_mask = inputs["attention_mask"].astype(np.int64)

# 4. Run inference multiple times and measure total time
NUM_RUNS = 10000
start_time = time.time()

for _ in tqdm(range(NUM_RUNS), desc="ONNX Inference"):
    outputs = sess.run(
        output_names=["logits"],
        input_feed={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / NUM_RUNS

# 5. Report
print(f"✅ 총 시간: {total_time:.4f}초")
#print(f"⏱️ 평균 추론 시간: {avg_time * 1000:.4f} ms")
