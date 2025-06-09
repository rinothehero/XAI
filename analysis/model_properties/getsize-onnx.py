import os

onnx_path = "shrunk_bert_state.pt"  # 또는 파일 경로 전체
file_size_bytes = os.path.getsize(onnx_path)
file_size_mb = file_size_bytes / (1024 * 1024)  # MB 단위로 변환

print(f"📦 ONNX 파일 크기: {file_size_mb:.2f} MB")
