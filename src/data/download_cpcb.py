import sys
import zipfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_DIR, KAGGLE_KEY


def download_from_kaggle(dataset: str, output_subdir: str = None):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] kaggle package not installed. Run: pip install kaggle")
        return False

    output_dir = RAW_DIR / "cpcb" / (output_subdir or dataset.split("/")[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CPCB] Downloading: {dataset} -> {output_dir.relative_to(RAW_DIR)}")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(output_dir), unzip=True)

    files = list(output_dir.rglob("*.csv"))
    print(f"[CPCB] {len(files)} CSV(s) extracted:")
    for f in files[:10]:
        size_kb = f.stat().st_size // 1024
        print(f"  - {f.name}  ({size_kb:,} KB)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    return True


def main():

    print("  CPCB Data Download")

    results = {}


    print("\n[1/2] Delhi hourly aggregate (deepaksirohiwal)...")
    results["hourly"] = download_from_kaggle(
        "deepaksirohiwal/delhi-air-quality",
        output_subdir="hourly",
    )


    print("\n[2/2] India city-level for NCR spatial coverage (rohanrao)...")
    results["spatial"] = download_from_kaggle(
        "rohanrao/air-quality-data-in-india",
        output_subdir="spatial",
    )


    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")

    if not all(results.values()):
        print("\n[WARN] Some downloads failed. Manual download:")
        print("  kaggle datasets download -d deepaksirohiwal/delhi-air-quality")
        print("  kaggle datasets download -d rohanrao/air-quality-data-in-india")
        print(f"  Extract both to: {RAW_DIR / 'cpcb'}")

    return all(results.values())


if __name__ == "__main__":
    main()
