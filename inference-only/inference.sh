python3 cpu-mem-monitor-8bit.py & echo $! > cpu_monitor-8bit.pid
python3 inference-quantized-8bit.py
kill $(cat cpu_monitor-8bit.pid)

#python3 gpu-mem-monitor.py &
#python3 inference-quantized-16bit.py
#sleep 3
#python3 inference-original.py
#sleep 3
#python3 inference-pruned-20.py
#sleep 3
#ython3 inference-pruned-40.py
#sleep 3
#python3 inference-quantized-pruned-16bit.py

python3 cpu-mem-monitor-pruned-8bit.py & echo $! > cpu_monitor-pruned-8bit.pid
python3 inference-quantized-pruned-8bit.py
kill $(cat cpu_monitor-pruned-8bit.pid)
