import subprocess
import time
import csv
from datetime import datetime

# 측정 간격 (초)
interval = 0.1
# 측정 시간 (초) – 무한 반복 원하면 None 설정
total_duration = 60  # 예: 10초간 측정
end_time = time.time() + total_duration if total_duration else None

# 출력 CSV 파일
output_file = "gpu_memory_log.csv"

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "memory.used (MiB)", "memory.total (MiB)"])

    while True:
        result = subprocess.run(
            ["nvidia-smi", "--id=1", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            text=True
        )
        mem_line = result.stdout.strip().split("\n")[0]  # 첫 번째 GPU 정보만 사용
        mem_used, mem_total = mem_line.split(", ")
        writer.writerow([mem_line, mem_used, mem_total])
        f.flush()  # 실시간 기록

        time.sleep(interval)
        if end_time and time.time() > end_time:
            break

print(f"✅ 로그 저장 완료: {output_file}")
