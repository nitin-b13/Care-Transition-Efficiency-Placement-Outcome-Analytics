"""Runs Modules 1-5 in order. Then start the dashboard with:  streamlit run app.py"""
import subprocess, sys
from pathlib import Path
here = Path(__file__).parent
for m in ["module1_cleaning", "module2_features", "module3_eda", "module4_kpis", "module5_bottleneck_model"]:
    print(f"\n=== {m} ===", flush=True)
    r = subprocess.run([sys.executable, str(here / f"{m}.py")], cwd=here)
    if r.returncode: sys.exit(f"{m} failed - read the error above")
print("\nAll modules finished. Now run:  streamlit run app.py")
