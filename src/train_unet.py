import gc
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

from src.datasets.ocean_dataset import OceanDataset
from src.models.unet2d import UNet2D
from src.models.physics_loss import PhysicsInformedLoss


def load_config(config_path: str = "configs/train_config.json") -> dict:
    path = Path(config_path)
    with open(path, "r") as f:
        return json.load(f)


def main():
    config = load_config()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    train_dataset = OceanDataset(nc_path=config["train_nc"])
    val_dataset = OceanDataset(nc_path=config["val_nc"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=2, persistent_workers=True
    )

    model = UNet2D().to(device)

    criterion = PhysicsInformedLoss(lambda_bulk=0.1, lambda_grad=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    epochs = 20
    best_val_loss = float("inf")
    use_amp = device.type in ["mps", "cuda"]

    print(f"Starting 2D U-Net Baseline Training ({epochs} Epochs)")

    for epoch in range(6, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                preds = model(x)
                loss, loss_dict = criterion(preds, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            l_tot = (
                loss_dict["loss_total"].item()
                if isinstance(loss_dict["loss_total"], torch.Tensor)
                else float(loss_dict["loss_total"])
            )
            train_loss += l_tot

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    preds = model(x)
                    loss, loss_dict = criterion(preds, y)

                l_tot = (
                    loss_dict["loss_total"].item()
                    if isinstance(loss_dict["loss_total"], torch.Tensor)
                    else float(loss_dict["loss_total"])
                )
                val_loss += l_tot

        val_loss /= len(val_loader)
        elapsed = time.strftime("%M:%S", time.gmtime(time.time() - epoch_start))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed}) | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = checkpoint_dir / "best_unet_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                save_path,
            )
            print(f"Saved best baseline model checkpoint to {save_path}")

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    print(
        "\nBaseline training complete! Checkpoint saved to checkpoints/best_unet_model.pt"
    )


if __name__ == "__main__":
    main()
