from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    eval_path = Path("data/eval_results/eval_outputs.npz")
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation file at {eval_path}. Run `python -m src.eval` first."
        )

    print(f"Loading evaluation arrays from {eval_path}...")
    data = np.load(eval_path)
    targets = data["targets"]  # Shape: (N, C, H, W)
    swin_preds = data["swin_preds"]  # Shape: (N, C, H, W)
    unet_preds = data["unet_preds"]  # Shape: (N, C, H, W)

    # Pick a target timestep (e.g., sample index 0) and channel 0 (Sea Surface Height / SSH)
    sample_idx = 0
    ch_idx = 0

    gt_ssh = targets[sample_idx, ch_idx]
    swin_ssh = swin_preds[sample_idx, ch_idx]
    unet_ssh = unet_preds[sample_idx, ch_idx]

    vmin = min(gt_ssh.min(), swin_ssh.min(), unet_ssh.min())
    vmax = max(gt_ssh.max(), swin_ssh.max(), unet_ssh.max())

    # Spatial SSH Forecast Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    im0 = axes[0].imshow(gt_ssh, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("Ground Truth (2024 Test)", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="SSH")

    im1 = axes[1].imshow(unet_ssh, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title("2D U-Net Baseline", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="SSH")

    im2 = axes[2].imshow(swin_ssh, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title("3D Swin-Transformer", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="SSH")

    plt.tight_layout()
    fig1_path = output_dir / "ssh_spatial_comparison.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Absolute Error Differential Maps
    unet_error = np.abs(gt_ssh - unet_ssh)
    swin_error = np.abs(gt_ssh - swin_ssh)
    error_vmax = max(unet_error.max(), swin_error.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    im_u = axes[0].imshow(
        unet_error, cmap="Reds", vmin=0, vmax=error_vmax, origin="lower"
    )
    axes[0].set_title("2D U-Net Absolute Error", fontsize=12, fontweight="bold")
    plt.colorbar(im_u, ax=axes[0], fraction=0.046, pad=0.04, label="|Error|")

    im_s = axes[1].imshow(
        swin_error, cmap="Reds", vmin=0, vmax=error_vmax, origin="lower"
    )
    axes[1].set_title(
        "3D Swin-Transformer Absolute Error", fontsize=12, fontweight="bold"
    )
    plt.colorbar(im_s, ax=axes[1], fraction=0.046, pad=0.04, label="|Error|")

    plt.tight_layout()
    fig2_path = output_dir / "error_differential_heatmaps.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig2_path}")

    # Mean Spatial Error Map (Averaged over all test timesteps)
    mean_swin_err = np.mean((targets[:, ch_idx] - swin_preds[:, ch_idx]) ** 2, axis=0)
    mean_unet_err = np.mean((targets[:, ch_idx] - unet_preds[:, ch_idx]) ** 2, axis=0)
    diff_map = (
        mean_unet_err - mean_swin_err
    )  # Positive values = Swin outperformed U-Net

    plt.figure(figsize=(8, 6))
    max_bound = np.max(np.abs(diff_map))
    plt.imshow(
        diff_map, cmap="coolwarm", vmin=-max_bound, vmax=max_bound, origin="lower"
    )
    plt.colorbar(label="MSE Advantage (U-Net MSE - Swin MSE)")
    plt.title(
        "Mean Spatial Loss Differential (Blue = Swin Superior)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    fig3_path = output_dir / "spatial_loss_differential.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig3_path}\n Visualization pipeline completed successfully.")


if __name__ == "__main__":
    main()
