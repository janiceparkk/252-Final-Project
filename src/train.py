import gc
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.datasets.ocean_dataset import OceanDataset
from src.models.swin_transformer import SwinTransformer3D
from src.models.physics_loss import PhysicsInformedLoss


def load_config(config_path: str = "configs/train_config.json") -> dict:
    path = Path(config_path)
    with open(path, "r") as f:
        return json.load(f)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs,
    accum_steps: int = 2,
    log_interval: int = 10,
):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_bulk = 0.0
    running_grad = 0.0

    total_batches = len(dataloader)
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    use_amp = device.type in ["mps", "cuda"]

    for step, (x, y) in enumerate(dataloader, start=1):
        x, y = x.to(device), y.to(device)

        # Forward pass
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            preds = model(x)
            loss, loss_dict = criterion(preds, y)
            scaled_loss = loss / accum_steps

        # Backward pass
        scaled_loss.backward()

        if step % accum_steps == 0 or step == total_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        l_tot = (
            loss_dict["loss_total"].item()
            if isinstance(loss_dict["loss_total"], torch.Tensor)
            else float(loss_dict["loss_total"])
        )
        l_mse = (
            loss_dict["loss_mse"].item()
            if isinstance(loss_dict["loss_mse"], torch.Tensor)
            else float(loss_dict["loss_mse"])
        )
        l_blk = (
            loss_dict["loss_bulk"].item()
            if isinstance(loss_dict["loss_bulk"], torch.Tensor)
            else float(loss_dict["loss_bulk"])
        )
        l_grd = (
            loss_dict["loss_grad"].item()
            if isinstance(loss_dict["loss_grad"], torch.Tensor)
            else float(loss_dict["loss_grad"])
        )

        running_loss += l_tot
        running_mse += l_mse
        running_bulk += l_blk
        running_grad += l_grd

        if step % log_interval == 0 or step == total_batches:
            elapsed = time.time() - start_time
            sec_per_batch = elapsed / step
            print(
                f"Epoch [{epoch:02d}/{total_epochs:02d}] "
                f"Batch [{step:03d}/{total_batches:03d}] "
                f"Loss: {l_tot:.4f} (MSE: {l_mse:.4f}) "
                f"Speed: {sec_per_batch:.2f}s/batch"
            )

        if step % 20 == 0 and device.type == "mps":
            torch.mps.empty_cache()

    return {
        "loss": running_loss / total_batches,
        "mse": running_mse / total_batches,
        "bulk": running_bulk / total_batches,
        "grad": running_grad / total_batches,
    }


def validate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    use_amp = device.type in ["mps", "cuda"]

    with torch.no_grad():
        for x, y in dataloader:
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
            l_mse = (
                loss_dict["loss_mse"].item()
                if isinstance(loss_dict["loss_mse"], torch.Tensor)
                else float(loss_dict["loss_mse"])
            )

            running_loss += l_tot
            running_mse += l_mse

    num_batches = len(dataloader)
    return {
        "loss": running_loss / num_batches,
        "mse": running_mse / num_batches,
    }


def main():
    config = load_config()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    print("Initializing Datasets and DataLoaders...")
    train_dataset = OceanDataset(nc_path=config["train_nc"])
    val_dataset = OceanDataset(nc_path=config["val_nc"])

    num_workers = config.get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = SwinTransformer3D(
        in_chans=config["in_chans"],
        out_chans=config["out_chans"],
        in_len=config["in_len"],
        out_len=config["out_len"],
        embed_dim=config["embed_dim"],
    ).to(device)

    criterion = PhysicsInformedLoss(
        lambda_bulk=config["lambda_bulk"], lambda_grad=config["lambda_grad"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=config["weight_decay"],
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=config["warmup_epochs"]
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=config["epochs"] - config["warmup_epochs"], eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config["warmup_epochs"]],
    )

    best_val_loss = float("inf")
    accum_steps = config.get("accum_steps", 2)
    print(f"Starting Optimized Swin-Transformer Training ({config['epochs']} Epochs)")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            config["epochs"],
            accum_steps=accum_steps,
            log_interval=10,
        )
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.strftime("%M:%S", time.gmtime(time.time() - epoch_start))

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:02d}/{config['epochs']:02d}] Summary ({elapsed}) "
            f"Train Loss: {train_metrics['loss']:.5f} "
            f"Val Loss: {val_metrics['loss']:.5f} "
            f"LR: {current_lr:.6f}\n"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = checkpoint_dir / "best_swin_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                },
                save_path,
            )
            print(f"Saved best model checkpoint to {save_path}")

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
