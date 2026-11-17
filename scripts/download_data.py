from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readmission.data import download_data

if __name__ == "__main__":
    print(download_data(ROOT / "data" / "raw"))
