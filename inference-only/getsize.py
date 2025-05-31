import os
import csv

def get_file_size_mb(file_path):
    """단일 .pt 파일 크기 계산"""
    return os.path.getsize(file_path) / (1024 * 1024)

def get_dir_size_mb(dir_path):
    """디렉토리 전체 크기 계산 (save_pretrained 방식)"""
    total = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)

def get_selected_files_size_mb(file_paths):
    total = 0
    for file_path in file_paths:
        if os.path.isfile(file_path):
            total += os.path.getsize(file_path)
        else:
            print(f"⚠️ 무시됨 (유효하지 않은 경로): {file_path}")
    return total / (1024 * 1024)

# ✅ 측정할 모델 파일들 (딱 2개만)
models_to_check = [
    {
        "model_name": "pruned_bert_agnews_structured-20-6",
        "files": [
            "../pruned_bert_agnews_structured-20-6/config.json",
            "../pruned_bert_agnews_structured-20-6/model.safetensors"
        ]
    },
    {
        "model_name": "pruned_bert_agnews_structured-30-6",
        "files": [
            "../pruned_bert_agnews_structured-30-6/config.json",
            "../pruned_bert_agnews_structured-30-6/model.safetensors"
        ]
    },
    {
        "model_name": "ft_BERT_agnews_full_dataset",
        "files": {
            "../models/ft_BERT_agnews_full_dataset/checkpoint-5064/config.json",
            "../models/ft_BERT_agnews_full_dataset/checkpoint-5064/model.safetensors"
        }
    }
]
# ✅ 측정할 모델 경로들 (.pt 파일 또는 디렉토리)
model_paths = [
    "../quantized-bert/quantized_bert_agnews_dynamic.pt",
    "../quantized-bert/quantized_bert_agnews_16bit.pt",
    #"../pruned_bert_agnews_structured-20/checkpoint-10125/config.json",
    #"../pruned_bert_agnews_structured-20/checkpoint-10125/model.safetensors",
    #"../pruned_bert_agnews_structured-30/checkpoint-10125/config.json",
    #"../pruned_bert_agnews_structured-30/checkpoint-10125/model.safetensors",
    "../quantized-bert/quantized_pruned_dynamic.pt",
    "../quantized-bert/quantized16_pruned_dynamic.pt",
]

# ✅ 결과 저장용 CSV
output_csv = "model_size_summary.csv"

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model_name", "type", "size_MB"])
    
    for model in models_to_check:
        size = get_selected_files_size_mb(model["files"])
        writer.writerow([model["model_name"], f"{size:.2f}"])
    
    for path in model_paths:
        if os.path.isfile(path) and path.endswith(".pt"):
            size = get_file_size_mb(path)
            model_type = "pt (quantized)"
        elif os.path.isdir(path):
            size = get_dir_size_mb(path)
            model_type = "directory (save_pretrained)"
        else:
            print(f"⚠️ 무시됨 (유효하지 않은 경로): {path}")
            continue

        model_name = os.path.basename(path)
        writer.writerow([model_name, model_type, f"{size:.2f}"])

print(f"✅ 결과 저장 완료: {output_csv}")
