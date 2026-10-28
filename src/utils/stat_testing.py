import numpy as np
import torch
from scipy import stats


def autocovariance(d: np.ndarray, d_bar: float, k: int):
    """
    Computes sample autocovariance at lag k.
    """
    T = len(d)
    k_int = int(k)
    if k_int >= T:
        return 0.0
    return float(np.sum((d[k_int:] - d_bar) * (d[: T - k_int] - d_bar)) / T)


def diebold_mariano_test(preds_a, preds_b, targets, h: int = 1):
    """
    Diebold-Mariano test comparing time-series forecast accuracy.
    Accepts 1D vectors or multi-dimensional spatial prediction tensors.
    """
    pa = (
        preds_a.detach().cpu().numpy()
        if isinstance(preds_a, torch.Tensor)
        else np.asarray(preds_a)
    )
    pb = (
        preds_b.detach().cpu().numpy()
        if isinstance(preds_b, torch.Tensor)
        else np.asarray(preds_b)
    )
    tg = (
        targets.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets)
    )

    # Compute mean loss per timestep across spatial and channel dimensions
    if pa.ndim > 1:
        spatial_axes = tuple(range(1, pa.ndim))
        loss_a = np.mean((tg - pa) ** 2, axis=spatial_axes)
        loss_b = np.mean((tg - pb) ** 2, axis=spatial_axes)
    else:
        loss_a = (tg - pa) ** 2
        loss_b = (tg - pb) ** 2

    # Loss differential time-series
    d = loss_a - loss_b
    T = len(d)
    d_bar = float(np.mean(d))

    h_int = int(h)

    if h_int <= 1:
        gamma_0 = autocovariance(d, d_bar, 0)
        var_d = gamma_0 / T
    else:
        gamma_0 = autocovariance(d, d_bar, 0)
        sum_gamma = sum(autocovariance(d, d_bar, k) for k in range(1, h_int))
        var_d = (gamma_0 + 2 * sum_gamma) / T

    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = float(d_bar / np.sqrt(var_d))
    p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))

    return dm_stat, p_value


def spatial_block_bootstrap(
    metric_a,
    metric_b,
    block_size: tuple[int, int] = (60, 60),
    num_iterations: int = 1000,
    seed: int = 42,
):
    """
    Executes spatial block bootstrapping on 2D error maps [H, W].
    Returns empirical bootstrapped p-value and 95% confidence interval.
    """
    ma = (
        metric_a.detach().cpu().numpy()
        if isinstance(metric_a, torch.Tensor)
        else np.asarray(metric_a)
    )
    mb = (
        metric_b.detach().cpu().numpy()
        if isinstance(metric_b, torch.Tensor)
        else np.asarray(metric_b)
    )

    # Reduce spatial metric maps to 2D grid (H, W) if necessary
    if ma.ndim > 2:
        ma = np.mean(ma, axis=tuple(range(0, ma.ndim - 2)))
    if mb.ndim > 2:
        mb = np.mean(mb, axis=tuple(range(0, mb.ndim - 2)))

    diff_map = ma - mb
    H, W = diff_map.shape
    bh, bw = int(block_size[0]), int(block_size[1])

    np.random.seed(seed)

    n_blocks_h = max(1, H // bh)
    n_blocks_w = max(1, W // bw)

    blocks = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            r_start, r_end = i * bh, min((i + 1) * bh, H)
            c_start, c_end = j * bw, min((j + 1) * bw, W)
            blocks.append(diff_map[r_start:r_end, c_start:c_end])

    num_blocks = len(blocks)
    boot_means = np.zeros(num_iterations)

    for b in range(num_iterations):
        sampled_indices = np.random.choice(num_blocks, size=num_blocks, replace=True)
        boot_sample = np.concatenate([blocks[idx].ravel() for idx in sampled_indices])
        boot_means[b] = np.mean(boot_sample)

    orig_mean = float(np.mean(diff_map))
    p_boot = float(np.mean(np.abs(boot_means - orig_mean) >= np.abs(orig_mean)))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return p_boot, (ci_lower, ci_upper)
