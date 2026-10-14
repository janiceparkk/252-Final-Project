import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """
    Composite Loss Function:
        L_total = L_MSE + lambda1 * L_bulk + lambda2 * L_gradient
    """

    def __init__(
        self,
        lambda_bulk: float = 0.1,
        lambda_grad: float = 0.05,
        sst_idx: int = 1,
        ssh_idx: int = 0,
        t2m_idx: int = 4,
        u10_idx: int = 6,
        v10_idx: int = 7,
    ):
        super().__init__()
        self.lambda_bulk = lambda_bulk
        self.lambda_grad = lambda_grad

        # Variable index positions
        self.sst_idx = sst_idx
        self.ssh_idx = ssh_idx
        self.t2m_idx = t2m_idx
        self.u10_idx = u10_idx
        self.v10_idx = v10_idx

        # Physical constants for Bulk Formulas
        self.rho_a = 1.225  # Air density (kg/m^3)
        self.c_p = 1004.0  # Specific heat capacity of air (J/kg*K)
        self.C_H = 1.4e-3  # Transfer coefficient for sensible heat
        self.L_v = 2.5e6  # Latent heat of vaporization (J/kg)
        self.C_E = 1.4e-3  # Transfer coefficient for latent heat

        self.mse_loss = nn.MSELoss()

    def compute_bulk_heat_flux(self, sst, t2m, u10, v10):
        """
        Computes sensible (QS) heat exchange between sea surface and atmosphere.
        """
        wind_speed = torch.sqrt(u10**2 + v10**2 + 1e-6)
        q_s = self.rho_a * self.c_p * self.C_H * wind_speed * (sst - t2m)
        return q_s

    def compute_spatial_gradient(self, ssh):
        """Computes spatial gradients along height (y) and width (x) dimensions."""

        # ssh shape: [B, 1, T, H, W] or [B, T, H, W]
        grad_y = torch.abs(ssh[..., 1:, :] - ssh[..., :-1, :])
        grad_x = torch.abs(ssh[..., :, 1:] - ssh[..., :, :-1])
        return grad_y, grad_x

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted tensor of shape [B, C, T, H, W]
            target: Ground truth tensor of shape [B, C, T, H, W]
        """
        # Base MSE Loss
        l_mse = self.mse_loss(pred, target)

        # Thermodynamic Bulk Formula Loss (L_bulk)
        pred_sst = pred[:, self.sst_idx]
        true_sst = target[:, self.sst_idx]
        true_t2m = target[:, self.t2m_idx]
        true_u10 = target[:, self.u10_idx]
        true_v10 = target[:, self.v10_idx]

        q_s_pred = self.compute_bulk_heat_flux(pred_sst, true_t2m, true_u10, true_v10)
        q_s_true = self.compute_bulk_heat_flux(true_sst, true_t2m, true_u10, true_v10)

        l_bulk = torch.mean(torch.abs(q_s_pred - q_s_true))

        # Spatial Gradient Loss on Sea Surface Height (L_gradient)
        pred_ssh = pred[:, self.ssh_idx]
        true_ssh = target[:, self.ssh_idx]

        pred_gy, pred_gx = self.compute_spatial_gradient(pred_ssh)
        true_gy, true_gx = self.compute_spatial_gradient(true_ssh)

        l_grad = torch.mean(torch.abs(pred_gy - true_gy)) + torch.mean(
            torch.abs(pred_gx - true_gx)
        )

        # Total Composite Loss
        l_total = l_mse + (self.lambda_bulk * l_bulk) + (self.lambda_grad * l_grad)

        loss_dict = {
            "loss_total": l_total.item(),
            "loss_mse": l_mse.item(),
            "loss_bulk": l_bulk.item(),
            "loss_grad": l_grad.item(),
        }

        return l_total, loss_dict
