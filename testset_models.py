from transformers import BertTokenizer
from lime.lime_text import LimeTextExplainer
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

# 1. Load tokenizer
model_path = "./models/ft_BERT_agnews_full_dataset"  # 학습 시 사용한 토크나이저 경로
tokenizer = BertTokenizer.from_pretrained(model_path)

# 2. Load pruned model
model = AutoModelForSequenceClassification.from_pretrained("./pruned_bert_agnews_structured-20", local_files_only=True)
model.to('cuda').eval()

# 3. Define class names
class_names = ["World", "Sports", "Business", "Sci/Tech"]

print("evaluating pruned 20% model")

