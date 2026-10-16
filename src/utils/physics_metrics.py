import numpy as np
import torch


def compute_rmse(pred, target, dim=(-2, -1)):
    """
    Args:
        pred: Predicted grid tensor [..., H, W]
        target: Ground truth grid tensor [..., H, W]
        dim: Spatial dimensions over which to compute mean square error
    Returns:
        Tensor of spatial RMSE values.
    """
    return torch.sqrt(torch.mean((target - pred) ** 2, dim=dim))


def compute_pss(rmse_model, rmse_persistence, eps=1e-8):
    """
    Computes Persistence Forecast Skill Score
    Values > 0 are evidence of better predictive skill than the baseline.
    """
    return 1.0 - (rmse_model / (rmse_persistence + eps))


def compute_eke(
    u,
    v,
    u_mean=None,
    v_mean=None,
):
    """
    Computes Eddy Kinetic Energy (EKE) where u' and v' are anomalies.
    """
    if not u_mean:
        u_mean = torch.mean(u, dim=(-2, -1), keepdim=True)
    if not v_mean:
        v_mean = torch.mean(v, dim=(-2, -1), keepdim=True)

    u_prime = u - u_mean
    v_prime = v - v_mean

    return 0.5 * (u_prime**2 + v_prime**2)


def compute_crps_gaussian(
    pred_mean,
    pred_std,
    target,
    eps=1e-6,
):
    """
    Computes Continuous Ranked Probability Score
    under a Gaussian uncertainty assumption.
    """
    scale = torch.clamp(pred_std, min=eps)
    z = (target - pred_mean) / scale

    norm = torch.distributions.Normal(0, 1)
    cdf = norm.cdf(z)
    pdf = torch.exp(norm.log_prob(z))

    crps = scale * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
    return crps
