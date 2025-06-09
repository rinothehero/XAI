python3 analysis/monitoring/cpu-mem-monitor-8bit.py & echo $! > analysis/monitoring/cpu_monitor-8bit.pid
#python3 inference-quantized-8bit.py
# TODO: Replace with call to run_inference_on_model.py for 8bit quantized model
kill $(cat analysis/monitoring/cpu_monitor-8bit.pid)

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

python3 analysis/monitoring/cpu-mem-monitor-pruned-8bit.py & echo $! > analysis/monitoring/cpu_monitor-pruned-8bit.pid
#python3 inference-quantized-pruned-8bit.py
# TODO: Replace with call to run_inference_on_model.py for 8bit pruned-quantized model
kill $(cat analysis/monitoring/cpu_monitor-pruned-8bit.pid)
