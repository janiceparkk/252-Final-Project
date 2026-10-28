from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.ocean_dataset import OceanDataset
from src.models.swin_transformer import SwinTransformer3D
from src.models.unet2d import UNet2D
from src.utils.physics_metrics import (
    compute_rmse,
    compute_pss,
    compute_eke,
    compute_crps_gaussian,
)
from src.utils.stat_testing import diebold_mariano_test, spatial_block_bootstrap


def ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    """Squeezes singleton dimensions until tensor is 4D (N, C, H, W)."""
    while tensor.ndim > 4:
        one_dims = [i for i, s in enumerate(tensor.shape) if s == 1]
        if one_dims:
            tensor = tensor.squeeze(one_dims[0])
        else:
            tensor = tensor[:, -1]
    return tensor


def run_inference(model, dataloader, device):
    """Executes FP16 evaluation pass and extracts predictions, targets, and persistence baseline."""
    model.eval()
    all_preds = []
    all_targets = []
    all_persistences = []
    use_amp = device.type in ["mps", "cuda"]

    with torch.no_grad():
        for x, y in dataloader:
            x_dev = x.to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                preds = model(x_dev)

            # Extract the last input time step as the persistence baseline
            if x.ndim == 5:
                if x.shape[1] == 7 or x.shape[1] == 1:  # (B, T, C, H, W)
                    persistence = x[:, -1, :, :, :]
                elif x.shape[2] == 7 or x.shape[2] == 1:  # (B, C, T, H, W)
                    persistence = x[:, :, -1, :, :]
                elif x.shape[1] == 8:  # (B, C, T, H, W) where C=8
                    persistence = x[:, :, -1, :, :]
                else:
                    persistence = x[:, -1, :, :, :]
            else:
                persistence = x

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_persistences.append(persistence.cpu())

    preds_tensor = torch.cat(all_preds, dim=0).float()
    targets_tensor = torch.cat(all_targets, dim=0).float()
    persist_tensor = torch.cat(all_persistences, dim=0).float()

    # Enforce uniform 4D tensor structure (N, C, H, W)
    preds_tensor = ensure_4d(preds_tensor)
    targets_tensor = ensure_4d(targets_tensor)
    persist_tensor = ensure_4d(persist_tensor)

    return preds_tensor, targets_tensor, persist_tensor


def main():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_nc = "data/processed/test_2024.nc"
    swin_ckpt_path = "checkpoints/best_swin_model.pt"
    unet_ckpt_path = "checkpoints/best_unet_model.pt"

    print("Loading 2024 Test Dataset...")
    test_dataset = OceanDataset(nc_path=test_nc)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    print(f"Running 3D Swin-Transformer ({swin_ckpt_path})...")
    swin_ckpt = torch.load(swin_ckpt_path, map_location=device)
    swin_config = swin_ckpt["config"]

    swin_model = SwinTransformer3D(
        in_chans=swin_config["in_chans"],
        out_chans=swin_config["out_chans"],
        in_len=swin_config["in_len"],
        out_len=swin_config["out_len"],
        embed_dim=swin_config["embed_dim"],
    ).to(device)
    swin_model.load_state_dict(swin_ckpt["model_state_dict"])

    swin_preds, targets, persist_preds = run_inference(swin_model, test_loader, device)

    print(f"Running 2D U-Net Baseline ({unet_ckpt_path})...")
    unet_ckpt = torch.load(unet_ckpt_path, map_location=device)

    unet_model = UNet2D().to(device)
    unet_model.load_state_dict(unet_ckpt["model_state_dict"])

    unet_preds, _, _ = run_inference(unet_model, test_loader, device)

    # Spatial RMSE [N, 8] -> scalar mean
    swin_rmse_map = compute_rmse(swin_preds, targets, dim=(-2, -1))
    unet_rmse_map = compute_rmse(unet_preds, targets, dim=(-2, -1))
    persist_rmse_map = compute_rmse(persist_preds, targets, dim=(-2, -1))

    swin_rmse = float(torch.mean(swin_rmse_map).item())
    unet_rmse = float(torch.mean(unet_rmse_map).item())

    # Pattern Synthetic Skill Score (PSS)
    swin_pss = float(torch.mean(compute_pss(swin_rmse_map, persist_rmse_map)).item())
    unet_pss = float(torch.mean(compute_pss(unet_rmse_map, persist_rmse_map)).item())

    # Eddy Kinetic Energy (EKE) Error (Channels 2 and 3 = u and v)
    u_gt, v_gt = targets[:, 2], targets[:, 3]
    u_swin, v_swin = swin_preds[:, 2], swin_preds[:, 3]
    u_unet, v_unet = unet_preds[:, 2], unet_preds[:, 3]

    eke_gt = compute_eke(u_gt, v_gt)
    eke_swin = compute_eke(u_swin, v_swin)
    eke_unet = compute_eke(u_unet, v_unet)

    swin_eke_err = float(torch.mean(torch.abs(eke_swin - eke_gt)).item())
    unet_eke_err = float(torch.mean(torch.abs(eke_unet - eke_gt)).item())

    # Continuous Ranked Probability Score (CRPS)
    swin_std = torch.std(targets - swin_preds, dim=0, keepdim=True) + 1e-6
    unet_std = torch.std(targets - unet_preds, dim=0, keepdim=True) + 1e-6

    swin_crps = float(
        torch.mean(compute_crps_gaussian(swin_preds, swin_std, targets)).item()
    )
    unet_crps = float(
        torch.mean(compute_crps_gaussian(unet_preds, unet_std, targets)).item()
    )

    print("\nMODEL PERFORMANCE COMPARISON")
    print(f"{'Metric':<35} | {'2D U-Net Baseline':<20} | {'3D Swin-Transformer':<20}")
    print("-" * 80)
    print(
        f"{'RMSE (Root Mean Square Error)':<35} | {unet_rmse:<20.4f} | {swin_rmse:<20.4f}"
    )
    print(
        f"{'PSS (Persistence Skill Score)':<35} | {unet_pss:<20.4f} | {swin_pss:<20.4f}"
    )
    print(
        f"{'CRPS (Gaussian Prob Score)':<35} | {unet_crps:<20.4f} | {swin_crps:<20.4f}"
    )
    print(
        f"{'EKE Error (Eddy Kinetic Energy)':<35} | {unet_eke_err:<20.4f} | {swin_eke_err:<20.4f}"
    )
    print("\nRunning Statistical Hypothesis Tests...")

    # Diebold-Mariano Test
    dm_stat, dm_pvalue = diebold_mariano_test(
        swin_preds.numpy(), unet_preds.numpy(), targets.numpy()
    )
    print(f"[Diebold-Mariano Test] DM Stat: {dm_stat:.4f} | p-value: {dm_pvalue:.4e}")

    # Spatial Block Bootstrapping
    swin_spatial_map = torch.mean((targets - swin_preds) ** 2, dim=(0, 1))
    unet_spatial_map = torch.mean((targets - unet_preds) ** 2, dim=(0, 1))

    p_boot, ci_95 = spatial_block_bootstrap(
        swin_spatial_map,
        unet_spatial_map,
        block_size=(60, 60),
        num_iterations=1000,
        seed=42,
    )
    print(f"[Spatial Block Bootstrap] Bootstrapped p-value: {p_boot:.4e}")
    print(
        f"[Spatial Block Bootstrap] 95% Confidence Interval: ({ci_95[0]:.4f}, {ci_95[1]:.4f})"
    )

    output_dir = Path("data/eval_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    np.savez_compressed(
        output_dir / "eval_outputs.npz",
        targets=targets.numpy(),
        swin_preds=swin_preds.numpy(),
        unet_preds=unet_preds.numpy(),
    )
    print(f"\nSaved evaluation arrays to {output_dir / 'eval_outputs.npz'}")


if __name__ == "__main__":
    main()
