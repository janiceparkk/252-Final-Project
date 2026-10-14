import torch
from torch.utils.data import DataLoader
from src.datasets.ocean_dataset import OceanDataset


def test_data_loader():
    train_file = "data/processed/train_2015_2022.nc"

    # Instantiate Dataset (7 days history -> 1 day forecast horizon)
    dataset = OceanDataset(nc_path=train_file)

    # Wrap in PyTorch DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Grab first mini-batch
    x_batch, y_batch = next(iter(loader))

    print("\n--- DataLoader Sanity Check ---")
    print(
        f"Input  Batch Shape (X): {x_batch.shape}  --> [Batch, Channels, T_in, Height, Width]"
    )
    print(
        f"Target Batch Shape (Y): {y_batch.shape}  --> [Batch, Channels, T_out, Height, Width]"
    )
    print(f"X Data Type: {x_batch.dtype}")
    print(f"Y Data Type: {y_batch.dtype}")
    print(f"Contains NaNs in X: {torch.isnan(x_batch).any().item()}")
    print(f"Contains NaNs in Y: {torch.isnan(y_batch).any().item()}")
    print("PyTorch Dataset verified successfully")


if __name__ == "__main__":
    test_data_loader()
