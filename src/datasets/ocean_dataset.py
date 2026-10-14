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
        stats_path,
        variables,
        in_len,
        out_len,
        target_variables,
    ):
        super().__init__()
        self.nc_path = Path(nc_path)
        self.stats_path = "data/stats/normalization_params.json"
        self.in_len = 7
        self.out_len = 1
        self.seq_len = in_len + out_len

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
        with open(stats_path, "r") as f:
            self.norm_stats = json.load(f)

        print(f"Loading {self.nc_path.name} into PyTorch Dataset...")
        ds = xr.open_dataset(self.nc_path)

        # Array shape: [Time, Channel, Lat, Lon]
        data_list = []
        for var in self.variables:
            arr = ds[var].values.astype(np.float32)

            mean = self.norm_stats[var]["mean"]
            std = self.norm_stats[var]["std"]

            # Z-score normalization
            arr_norm = (arr - mean) / (std + 1e-7)
            arr_norm = np.nan_to_num(arr_norm, nan=0.0)

            data_list.append(arr_norm)

        # Array shape: [Time, Channels, Height, Width]
        self.data = np.stack(data_list, axis=1)
        self.num_timesteps = self.data.shape[0]

        print(f"Loaded tensor shape: {self.data.shape} [T, C, H, W]")

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
        # [T_in, C, H, W]
        input_window = self.data[idx : idx + self.in_len]

        # [T_out, C, H, W]
        target_window = self.data[idx + self.in_len : idx + self.seq_len]

        # Rearrange to 3D Swin-Transformer format: [C, T, H, W]
        x = np.transpose(input_window, (1, 0, 2, 3))

        # Select targets and transpose to [C_out, T_out, H, W]
        y = target_window[:, self.target_indices, :, :]
        y = np.transpose(y, (1, 0, 2, 3))

        return torch.from_numpy(x), torch.from_numpy(y)
