import sys
from pathlib import Path
import calendar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_DIR, DELHI_BBOX, YEARS


def download_era5_month(year: int, month: int, output_dir: Path):
    import cdsapi

    output_file = output_dir / f"era5_delhi_{year}_{month:02d}.nc"
    if output_file.exists():
        print(f"[ERA5] {output_file.name} already exists, skipping.")
        return output_file


    _, n_days = calendar.monthrange(year, month)
    days = [f"{d:02d}" for d in range(1, n_days + 1)]

    print(f"[ERA5] Requesting {year}-{month:02d} ({n_days} days)...")

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "boundary_layer_height",
            ],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": DELHI_BBOX,
            "format": "netcdf",
        },
        str(output_file),
    )

    size_mb = output_file.stat().st_size / 1e6
    print(f"[ERA5] Saved: {output_file.name} ({size_mb:.1f} MB)")
    return output_file


def main():

    print("  ERA5 Wind Data Download (monthly batches)")

    output_dir = RAW_DIR / "era5"
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for year in YEARS:
        for month in range(1, 13):
            try:
                download_era5_month(year, month, output_dir)
            except Exception as e:
                print(f"[ERA5] Failed for {year}-{month:02d}: {e}")
                print("  Make sure ~/.cdsapirc is configured with your CDS API key.")
                print("  Register at: https://cds.climate.copernicus.eu")
                failed.append((year, month))

    if failed:
        print(f"\n[ERA5] {len(failed)} months failed: {failed}")
    else:
        print("\n[ERA5] All months downloaded successfully.")


if __name__ == "__main__":
    main()
