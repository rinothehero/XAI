# cpu_mem_monitor.py

import psutil
import os
import time
import csv
from datetime import datetime

interval = 0.1  # 측정 간격 (초)
pid = os.getpid()
process = psutil.Process(pid)

output_file = "cpu_memory_log-pruned-8bit.csv"
duration = 300  # 총 측정 시간 (초). None으로 설정하면 무한

end_time = time.time() + duration if duration else None

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "memory_used_MB"])

    while True:
        mem = process.memory_info().rss / (1024 * 1024)  # MB
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        writer.writerow([timestamp, mem])
        f.flush()
        time.sleep(interval)
        if end_time and time.time() > end_time:
            break
