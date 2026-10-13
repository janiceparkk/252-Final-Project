import os
import json
from pathlib import Path
from dotenv import load_dotenv

import cdsapi
import copernicusmarine

load_dotenv()
CONFIG_PATH = Path("../configs/config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def download_glorys(config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "glorys_gulfstream_2015_2024.nc"

    if output_file.exists():
        return

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["zos", "thetao", "uo", "vo"],
        minimum_longitude=config["spatial"]["lon_min"],
        maximum_longitude=config["spatial"]["lon_max"],
        minimum_latitude=config["spatial"]["lat_min"],
        maximum_latitude=config["spatial"]["lat_max"],
        start_datetime=f"{config['temporal']['start_date']}T00:00:00",
        end_datetime=f"{config['temporal']['end_date']}T23:59:59",
        minimum_depth=config["depth"]["depth_min_m"],
        maximum_depth=config["depth"]["depth_max_m"],
        output_filename=output_file.name,
        output_directory=str(output_dir),
        username=os.getenv("COPERNICUSMARINE_USERNAME"),
        password=os.getenv("COPERNICUSMARINE_PASSWORD"),
        force_download=True,
    )


def download_era5(config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client(url=os.getenv("CDSAPI_URL"), key=os.getenv("CDSAPI_KEY"))

    start_year = int(config["temporal"]["start_date"].split("-")[0])
    end_year = int(config["temporal"]["end_date"].split("-")[0])

    for year in range(start_year, end_year + 1):
        year_str = str(year)
        for month in range(1, 13):
            month_str = f"{month:02d}"
            output_file = output_dir / f"era5_gulfstream_{year_str}_{month_str}.nc"

            if output_file.exists():
                continue

            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        "2m_temperature",
                        "2m_dewpoint_temperature",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                    "year": year_str,
                    "month": month_str,
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(0, 24)],
                    "area": [
                        config["spatial"]["lat_max"],
                        config["spatial"]["lon_min"],
                        config["spatial"]["lat_min"],
                        config["spatial"]["lon_max"],
                    ],
                },
                str(output_file),
            )


if __name__ == "__main__":
    cfg = load_config()
    raw_data_path = Path("../data/raw")

    download_glorys(cfg, raw_data_path / "glorys12v4")
    download_era5(cfg, raw_data_path / "era5")
