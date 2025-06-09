import time
import psutil
import subprocess

def is_graph_construction_running():
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and 'graph_construction.py' in ' '.join(proc.info['cmdline']):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

print("🔍 graph_construction.py가 종료될 때까지 대기 중...")

while is_graph_construction_running():
    time.sleep(5)

print("✅ graph_construction.py 종료됨. json_to_gml.py 실행 중...")
subprocess.run(["python", "json_to_gml.py"])
