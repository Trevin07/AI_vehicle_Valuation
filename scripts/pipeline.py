import subprocess
import sys

files = [
    "joined.py",
    "data_pre.py",
    "filling.py",
    "final_violation_check.py",
    "map.py"
]

for file in files:
    result = subprocess.run([sys.executable, file])
    if result.returncode != 0:
        print(f"Pipeline stopped. Error in {file}")
        break
