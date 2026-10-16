import numpy as np
from scipy import stats
import torch


def autocovariance(d, d_bar, k):
    if k == 0:
        return float(np.mean((d - d_bar) ** 2))
    return float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))


def diebold_mariano_test(
    loss_a,
    loss_b,
    h=1,
):
    """
    Evaluates the null hypothesis

    Args:
        loss_a: Loss time series for Baseline Model
        loss_b: Loss time series for Proposed Model
        h: Forecast days

    Returns:
        dm_stat: Diebold-Mariano test statistic
        p_value: Two-tailed p-value
    """
    if isinstance(loss_a, torch.Tensor):
        loss_a = loss_a.detach().cpu().numpy()
    if isinstance(loss_b, torch.Tensor):
        loss_b = loss_b.detach().cpu().numpy()

    d = loss_a - loss_b
    T = len(d)
    d_bar = float(np.mean(d))

    gamma_0 = autocovariance(d, d_bar, 0)
    sum_gamma = sum(autocovariance(d, d_bar, k) for k in range(1, h))

    variance = (gamma_0 + 2.0 * sum_gamma) / T

    if variance <= 1e-12:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(variance)
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    return float(dm_stat), float(p_value)


def spatial_block_bootstrap(
    metric_a,
    metric_b,
    block_size=(60, 60),
    num_iterations=1000,
    seed=42,
):
    """
    Args:
        metric_a: Spatial metric map for Model A [..., H, W]
        metric_b: Spatial metric map for Model B [..., H, W]
        block_size: (block_h, block_w)
        num_iterations: Number of bootstrap sampling iterations
        seed: Random seed

    Returns:
        p_boot: Empirical bootstrapped p-value
        ci_95: 95% confidence interval tuple (CI_2.5%, CI_97.5%)
    """
    if isinstance(metric_a, torch.Tensor):
        metric_a = metric_a.detach().cpu().numpy()
    if isinstance(metric_b, torch.Tensor):
        metric_b = metric_b.detach().cpu().numpy()

    np.random.seed(seed)
    h, w = metric_a.shape[-2:]
    block_h, block_w = block_size

    n_blocks_h = h // block_h
    n_blocks_w = w // block_w

    blocks_a = []
    blocks_b = []

    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            r_start, r_end = i * block_h, (i + 1) * block_h
            c_start, c_end = j * block_w, (j + 1) * block_w

            blocks_a.append(metric_a[..., r_start:r_end, c_start:c_end])
            blocks_b.append(metric_b[..., r_start:r_end, c_start:c_end])

    num_blocks = len(blocks_a)
    diff_bootstrap = []

    for _ in range(num_iterations):
        indices = np.random.choice(num_blocks, size=num_blocks, replace=True)

        sample_a = np.mean([blocks_a[idx] for idx in indices])
        sample_b = np.mean([blocks_b[idx] for idx in indices])

        diff_bootstrap.append(sample_b - sample_a)

    diff_bootstrap = np.array(diff_bootstrap)

    # Empirical p-value: Fraction of samples where Model B failed to improve over Model A
    p_boot = float(np.mean(diff_bootstrap >= 0.0))

    ci_low = float(np.percentile(diff_bootstrap, 2.5))
    ci_high = float(np.percentile(diff_bootstrap, 97.5))

    return p_boot, (ci_low, ci_high)
