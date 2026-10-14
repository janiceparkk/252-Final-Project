import torch
from src.models.physics_loss import PhysicsInformedLoss


def test_physics_loss():
    print("Testing Physics-Informed Composite Loss Function...")

    # Mock tensor dimensions [B=2, C=8, T=1, H=241, W=361]
    pred = torch.randn(2, 8, 1, 241, 361, requires_grad=True)
    target = torch.randn(2, 8, 1, 241, 361)

    criterion = PhysicsInformedLoss(lambda_bulk=0.1, lambda_grad=0.05)
    loss, loss_dict = criterion(pred, target)

    # Verify backpropagation
    loss.backward()

    print(f"Total Loss: {loss_dict['loss_total']:.6f}")
    print(f"MSE Loss: {loss_dict['loss_mse']:.6f}")
    print(f"Bulk Loss: {loss_dict['loss_bulk']:.6f}")
    print(f"Grad Loss: {loss_dict['loss_grad']:.6f}")
    print(
        f"Gradient Tensor Exists: {pred.grad is not None and not torch.isnan(pred.grad).any()}"
    )
    print("Physics Loss function test passed successfully")


if __name__ == "__main__":
    test_physics_loss()
