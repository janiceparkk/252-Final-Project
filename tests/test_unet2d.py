import torch
from src.models.unet2d import UNet2D


def test_unet_forward():
    print("Testing 2D U-Net Baseline...")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Input dimensions
    b, c_in, t_in, h, w = 2, 8, 7, 241, 361
    c_out, t_out = 8, 1

    x = torch.randn(b, c_in, t_in, h, w, device=device)
    expected_shape = torch.Size([b, c_out, t_out, h, w])

    model = UNet2D(
        in_chans=c_in,
        out_chans=c_out,
        in_len=t_in,
        out_len=t_out,
        features=[64, 128, 256, 512],
    ).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(x)

    print(f"Input Tensor Shape: {x.shape}")
    print(f"Output Forecast Shape: {preds.shape}")
    print(f"Expected Target Shape: {expected_shape}")
    print(f"Match Target Shape: {preds.shape == expected_shape}")
    print(f"Contains NaNs: {torch.isnan(preds).any().item()}")

    assert (
        preds.shape == expected_shape
    ), f"Shape mismatch: {preds.shape} vs {expected_shape}"
    assert not torch.isnan(preds).any(), "Model produced NaN values"

    print("2D U-Net forward pass completed successfully")


if __name__ == "__main__":
    test_unet_forward()
