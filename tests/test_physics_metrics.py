import torch
from src.utils.physics_metrics import (
    compute_rmse,
    compute_pss,
    compute_eke,
    compute_crps_gaussian,
)


def test_physics_metrics():
    print("Testing Ocean Physics Evaluation Metrics...")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Mock spatial grids
    b, c, h, w = 2, 8, 241, 361
    pred = torch.randn(b, c, h, w, device=device)
    target = pred + torch.randn(b, c, h, w, device=device) * 0.1
    persistence_pred = pred + torch.randn(b, c, h, w, device=device) * 0.5

    # Test 1: RMSE
    rmse_model = compute_rmse(pred, target)
    rmse_zero = compute_rmse(pred, pred)
    assert rmse_model.shape == (b, c), f"RMSE shape mismatch: {rmse_model.shape}"
    assert torch.allclose(
        rmse_zero, torch.zeros_like(rmse_zero), atol=1e-6
    ), "RMSE for identical tensors should be zero"
    print(
        f"RMSE shape: {tuple(rmse_model.shape)}, Mean Value: {rmse_model.mean().item():.5f}"
    )

    # Test 2: Persistence Skill Score (PSS)
    rmse_pers = compute_rmse(persistence_pred, target)
    pss = compute_pss(rmse_model, rmse_pers)
    pss_perfect = compute_pss(torch.zeros_like(rmse_pers), rmse_pers)
    assert pss.shape == (b, c), f"PSS shape mismatch: {pss.shape}"
    assert torch.all(pss_perfect > 0.99), "PSS for zero error should approach 1.0"
    print(f"PSS shape: {tuple(pss.shape)}, Mean Value: {pss.mean().item():.5f}")

    # Test 3: Eddy Kinetic Energy (EKE)
    u = torch.randn(b, h, w, device=device)
    v = torch.randn(b, h, w, device=device)
    eke = compute_eke(u, v)
    assert eke.shape == (b, h, w), f"EKE shape mismatch: {eke.shape}"
    assert torch.all(eke >= 0), "EKE values must be non-negative"
    print(f"EKE shape: {tuple(eke.shape)}, Mean Value: {eke.mean().item():.5f}")

    # Test 4: Continuous Ranked Probability Score (CRPS)
    std = torch.ones_like(pred) * 0.1
    crps = compute_crps_gaussian(pred, std, target)
    assert crps.shape == (b, c, h, w), f"CRPS shape mismatch: {crps.shape}"
    assert torch.all(crps >= 0), "CRPS values must be non-negative"
    print(f"CRPS shape: {tuple(crps.shape)}, Mean Value: {crps.mean().item():.5f}")

    print("All physics metric tests completed successfully")


if __name__ == "__main__":
    test_physics_metrics()
