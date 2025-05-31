from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# 1. AG News 데이터셋 로드
dataset = load_dataset("ag_news")
# train/valid 분할 (90:10)
split = dataset["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
train = split["train"]
val = split["test"]
test = dataset["test"]  # 7,600개는 그대로 test로 유지

# 2. Tokenizer 준비
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# 4. 각각 토큰화 및 Tensor 형식 설정
encoded_train = train.map(tokenize_fn, batched=True)
encoded_val = val.map(tokenize_fn, batched=True)
encoded_test = test.map(tokenize_fn, batched=True)


for ds in [encoded_train, encoded_val, encoded_test]:
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4. 모델 준비 (num_labels=4: AG News는 4개 클래스)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# 5. 학습 설정
args = TrainingArguments(
    output_dir="./ft_BERT_agnews_full_dataset",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 6. 정확도 메트릭 설정
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# 7. Trainer 정의
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    compute_metrics=compute_metrics,
)

# 8. 학습 시작
trainer.train()

# 9. 모델 저장
trainer.save_model("./ft_BERT_agnews_full_dataset")
tokenizer.save_pretrained("./ft_BERT_agnews_full_dataset")
print("Fine-tuning complete. Model saved to './ft_BERT_agnews_full_dataset'")