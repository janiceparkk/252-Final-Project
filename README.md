# 252-Final-Project
Using Microsoft's Swin Transformer to forecast the weather and ocean conditions for the Gulf Stream.
This project uses the [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) and [GLORYS](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) datasets.

## Installing Dependencies
```bash
pip install "numpy<2.0" copernicusmarine cdsapi chardet python-dotenv xarray netCDF4 scipy dask torch torchvision torchaudio
```

## Downloading the Datasets
```python
python scripts/download_dataset.py
```

## Processing the Datasets
```python
python scripts/process_data.py
```

## Running a Mock Test
```python
python -m tests.REPLACE_WITH_TEST_NAME
```

## Running Swin-Transformer Training
```python
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 caffeinate -i "$(python -c 'import sys; print(sys.executable)')" -m src.train
```

## Running Baseline Training
```python
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 caffeinate -i "$(python -c 'import sys; print(sys.executable)')" -m src.train_unet
```