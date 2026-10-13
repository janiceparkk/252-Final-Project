import json
from pathlib import Path
import xarray as xr
import numpy as np


def process_data():
    """
    The ERA5 and GLORYS differ in temporal and spatial measurements.
    ERA5 uses hourly measurements in 30km grids.
    GLORYS uses the avg daily measurements in 8km grids.
    Processing requires normalizing into common measurements.
    """
    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim/era5_daily")
    processed_dir = Path("data/processed")
    stats_dir = Path("data/stats")

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    glorys_path = raw_dir / "glorys12v4" / "glorys_gulfstream_2015_2024.nc"

    # Chunk GLORYS to keep memory low
    ds_glorys = xr.open_dataset(glorys_path, chunks={"time": 30})

    if "latitude" in ds_glorys.coords:
        ds_glorys = ds_glorys.rename({"latitude": "lat", "longitude": "lon"})

    target_lat = ds_glorys.lat
    target_lon = ds_glorys.lon

    era5_raw_files = sorted(list((raw_dir / "era5").glob("era5_gulfstream_*.nc")))
    print(
        f"Convert {len(era5_raw_files)} ERA5 monthly files into daily intermediate files..."
    )

    for fpath in era5_raw_files:
        out_file = interim_dir / fpath.name
        if out_file.exists():
            continue

        print(f"Resampling & regridding: {fpath.name}")
        ds_month = xr.open_dataset(fpath)

        time_dim = (
            "valid_time"
            if ("valid_time" in ds_month.dims or "valid_time" in ds_month.coords)
            else "time"
        )

        if "latitude" in ds_month.coords:
            ds_month = ds_month.rename({"latitude": "lat", "longitude": "lon"})

        ds_daily = ds_month.resample({time_dim: "1D"}).mean()
        if time_dim != "time":
            ds_daily = ds_daily.rename({time_dim: "time"})

        ds_regrid = ds_daily.interp(lat=target_lat, lon=target_lon, method="linear")

        ds_regrid.to_netcdf(out_file)
        ds_month.close()

    print("Intermediate ERA5 files ready.")

    era5_interim_files = sorted(list(interim_dir.glob("era5_gulfstream_*.nc")))
    ds_era5_aligned = xr.open_mfdataset(
        era5_interim_files, combine="by_coords", chunks={"time": 30}
    )

    ds_era5_aligned["lat"] = ds_glorys["lat"]
    ds_era5_aligned["lon"] = ds_glorys["lon"]

    common_times = np.intersect1d(ds_glorys.time.values, ds_era5_aligned.time.values)
    ds_glorys = ds_glorys.sel(time=common_times)
    ds_era5_aligned = ds_era5_aligned.sel(time=common_times)

    print("Merging GLORYS and ERA5 datasets...")
    ds_merged = xr.merge([ds_glorys, ds_era5_aligned]).astype(np.float32)

    print("Splitting datasets...")
    train_ds = ds_merged.sel(time=slice("2015-01-01", "2022-12-31"))
    val_ds = ds_merged.sel(time=slice("2023-01-01", "2023-12-31"))
    test_ds = ds_merged.sel(time=slice("2024-01-01", "2024-12-31"))

    print("Normalizing statistics on Training set...")
    norm_stats = {}
    var_list = ["zos", "thetao", "uo", "vo", "t2m", "d2m", "u10", "v10"]

    for var in var_list:
        if var in train_ds:
            print(f"Computing mean & std for: {var}")
            mean_val = float(train_ds[var].mean().compute())
            std_val = float(train_ds[var].std().compute())
            norm_stats[var] = {"mean": mean_val, "std": std_val}

    stats_file = stats_dir / "normalization_params.json"
    with open(stats_file, "w") as f:
        json.dump(norm_stats, f, indent=4)
    print(f"Normalization parameters saved to {stats_file}")

    print("Writing train_2015_2022.nc...")
    train_ds.to_netcdf(processed_dir / "train_2015_2022.nc")

    print("Writing val_2023.nc...")
    val_ds.to_netcdf(processed_dir / "val_2023.nc")

    print("Writing test_2024.nc...")
    test_ds.to_netcdf(processed_dir / "test_2024.nc")

    print("Preprocessing complete.")


if __name__ == "__main__":
    process_data()
