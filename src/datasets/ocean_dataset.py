import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class OceanDataset(Dataset):
    """
    Takes in file paths, variables, the input time steps, and the forecasted time steps.
    Returns tensor windows for the 3D Swin Transformers after z-score normalization.
    """

    def __init__(
        self,
        nc_path,
        variables=None,
        target_variables=None,
    ):
        super().__init__()
        self.nc_path = Path(nc_path)
        self.stats_path = "data/stats/normalization_params.json"
        self.in_len = 7
        self.out_len = 1
        self.seq_len = self.in_len + self.out_len

        if not variables:
            self.variables = ["zos", "thetao", "uo", "vo", "t2m", "d2m", "u10", "v10"]
        else:
            self.variables = variables

        if not target_variables:
            self.target_variables = self.variables
        else:
            self.target_variables = target_variables

        self.target_indices = [self.variables.index(v) for v in self.target_variables]

        # Load normalization parameters
        with open(self.stats_path, "r") as f:
            self.norm_stats = json.load(f)

        self.ds = xr.open_dataset(self.nc_path)
        self.num_timesteps = len(self.ds.time)

        print(f"Initialized OceanDataset from {self.nc_path.name}")

    def __len__(self):
        """
        Returns the total number of usable samples
        """
        return self.num_timesteps - self.seq_len + 1

    def __getitem__(self, idx):
        """
        Returns:
            x: Input tensor of shape [C, T_in, H, W]
            y: Target tensor of shape [C_out, T_out, H, W]
        """
        # Extract temporal slice
        slice_ds = self.ds.isel(time=slice(idx, idx + self.seq_len))

        normalized_vars = []
        for var in self.variables:
            da = slice_ds[var]

            # Slice away extra dimensions (e.g. depth, level, surface)
            for dim_name in list(da.dims):
                if dim_name not in ["time", "lat", "lon"]:
                    da = da.isel({dim_name: 0})

            # Enforce explicit dimension ordering
            da = da.transpose("time", "lat", "lon")
            arr = da.values.astype(np.float32)

            mean = self.norm_stats[var]["mean"]
            std = self.norm_stats[var]["std"]

            # Z-score normalization
            arr_norm = (arr - mean) / (std + 1e-7)
            arr_norm = np.nan_to_num(arr_norm, nan=0.0)

            normalized_vars.append(arr_norm)

        # Stack into uniform shape: [seq_len, Channels, Height, Width]
        window_data = np.stack(normalized_vars, axis=1)

        # [T_in, C, H, W]
        input_window = window_data[: self.in_len]

        # [T_out, C_out, H, W]
        target_window = window_data[self.in_len :, self.target_indices]

        # 3D Swin format: [C, T, H, W]
        x = np.transpose(input_window, (1, 0, 2, 3))
        y = np.transpose(target_window, (1, 0, 2, 3))

        return torch.from_numpy(x), torch.from_numpy(y)
