import numpy as np
import torch
from src.utils.stat_testing import diebold_mariano_test, spatial_block_bootstrap


def test_stat_testing():
    print("Testing Statistical Hypothesis...")

    # Diebold-Mariano Test
    np.random.seed(42)
    t_days = 365

    # Case A: Model B is significantly better than Model A (lower loss)
    loss_a = np.random.normal(loc=0.5, scale=0.1, size=t_days)
    loss_b = np.random.normal(loc=0.3, scale=0.1, size=t_days)

    dm_stat, p_val = diebold_mariano_test(loss_a, loss_b, h=1)
    print(f"DM Statistic: {dm_stat:.4f}, p-value: {p_val:.6e}")
    assert dm_stat > 0, "DM stat should be positive when Model B is superior"
    assert p_val < 0.05, f"p-value should be < 0.05 for distinct models, got {p_val}"

    # Case B: Identical models (Null hypothesis holds)
    loss_identical = np.copy(loss_a)
    dm_null, p_null = diebold_mariano_test(loss_a, loss_identical, h=1)
    assert abs(dm_null) < 1e-5, "DM stat for identical losses should be ~0"
    assert p_null == 1.0, "p-value for identical losses should be 1.0"

    # Test Spatial Block Bootstrapping
    h, w = 241, 361
    metric_a = np.random.normal(loc=0.4, scale=0.05, size=(h, w))
    metric_b = np.random.normal(loc=0.2, scale=0.05, size=(h, w))

    p_boot, (ci_low, ci_high) = spatial_block_bootstrap(
        metric_a,
        metric_b,
        block_size=(60, 60),
        num_iterations=200,
        seed=42,
    )

    print(f"Bootstrapped p-value: {p_boot:.4f}")
    print(f"95% Confidence Interval: [{ci_low:.5f}, {ci_high:.5f}]")

    assert 0.0 <= p_boot <= 1.0, "p_boot must be in [0, 1]"
    assert ci_low < ci_high, "CI lower bound must be less than upper bound"

    print("Statistical testing verification passed successfully")


if __name__ == "__main__":
    test_stat_testing()
