"""Quick monitor to check training progress"""
import time
from pathlib import Path
import os

model_path = Path(__file__).parent / 'best_model.pth'

print("Monitoring exp041 training...")
print("Press Ctrl+C to stop\n")

last_size = 0
last_mtime = 0

try:
    while True:
        if model_path.exists():
            stat = os.stat(model_path)
            mtime = stat.st_mtime
            size = stat.st_size
            
            if mtime != last_mtime:
                mtime_str = time.strftime("%H:%M:%S", time.localtime(mtime))
                print(f"[{time.strftime('%H:%M:%S')}] Model updated at {mtime_str}, size: {size} bytes")
                last_mtime = mtime
                last_size = size
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for model...")
        
        time.sleep(30)
except KeyboardInterrupt:
    print("\nMonitoring stopped.")
