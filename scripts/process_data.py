import json
from pathlib import Path
import xarray as xr
import numpy as np

CONFIG_PATH = Path("../configs/config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def process_data():
    """
    The era5 and glorys datasets differ in temporal and spatial alignment.
    Era5 provides hourly metrics while glorys provides the daily mean.
    Era5 uses a 31 km grid while glorys uses an 8 km grid.
    Processing both datasets requires aligning the datasets into common measurements.
    """
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    stats_dir = Path("data/stats")

    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    glorys_path = raw_dir / "glorys12v4" / "glorys_gulfstream_2015_2024.nc"
    ds_glorys = xr.open_dataset(glorys_path)

    era5_files = sorted(list((raw_dir / "era5").glob("era5_gulfstream_*.nc")))
    ds_era5 = xr.open_mfdataset(era5_files, combine="by_coords")

    # Resample hourly era5 to daily mean
    ds_era5_daily = ds_era5.resample(valid_time="1D").mean()

    if "valid_time" in ds_era5_daily.dims:
        ds_era5_daily = ds_era5_daily.rename({"valid_time": "time"})

    common_times = np.intersect1d(ds_glorys.time.values, ds_era5_daily.time.values)
    ds_glorys = ds_glorys.sel(time=common_times)
    ds_era5_daily = ds_era5_daily.sel(time=common_times)

    # Regrid era5 to glorys dimensions
    if "latitude" in ds_era5_daily.coords:
        ds_era5_daily = ds_era5_daily.rename({"latitude": "lat", "longitude": "lon"})
    if "latitude" in ds_glorys.coords:
        ds_glorys = ds_glorys.rename({"latitude": "lat", "longitude": "lon"})

    ds_era5_aligned = ds_era5_daily.interp(
        lat=ds_glorys.lat, lon=ds_glorys.lon, method="linear"
    )

    # Combine
    ds_merged = xr.merge([ds_glorys, ds_era5_aligned])

    train_ds = ds_merged.sel(time=slice("2015-01-01", "2022-12-31"))
    val_ds = ds_merged.sel(time=slice("2023-01-01", "2023-12-31"))
    test_ds = ds_merged.sel(time=slice("2024-01-01", "2024-12-31"))

    # Z-score normalization
    norm_stats = {}
    var_list = ["zos", "thetao", "uo", "vo", "t2m", "d2m", "u10", "v10"]

    for var in var_list:
        if var in train_ds:
            mean_val = float(train_ds[var].mean().values)
            std_val = float(train_ds[var].std().values)
            norm_stats[var] = {"mean": mean_val, "std": std_val}

    stats_file = stats_dir / "normalized_params.json"
    with open(stats_file, "w") as f:
        json.dump(norm_stats, f, indent=4)

    train_ds.to_netcdf(processed_dir / "train_2015_2022.nc")
    val_ds.to_netcdf(processed_dir / "validation_2023.nc")
    test_ds.to_netcdf(processed_dir / "test_2024.nc")


if __name__ == "__main__":
    process_data()
