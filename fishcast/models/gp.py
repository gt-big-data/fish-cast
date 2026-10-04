from __future__ import annotations

from typing import Dict

import torch
from torch import nn

try:
    import gpytorch
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.kernels import Kernel, ScaleKernel
    from gpytorch.means import ZeroMean
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "gpytorch is required for fishcast.models.gp. Activate the 'krillcast' venv or install gpytorch."
    ) from exc


class CovarianceNetwork(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, spatial_coords: torch.Tensor) -> torch.Tensor:
        raw = self.net(spatial_coords)
        diag_x = torch.nn.functional.softplus(raw[..., 0]) + 1e-3
        diag_y = torch.nn.functional.softplus(raw[..., 2]) + 1e-3
        lower = raw[..., 1]

        chol = torch.zeros(*raw.shape[:-1], 2, 2, device=raw.device, dtype=raw.dtype)
        chol[..., 0, 0] = diag_x
        chol[..., 1, 0] = lower
        chol[..., 1, 1] = diag_y
        return chol @ chol.transpose(-1, -2)


class DeepNonStationaryKernel(Kernel):
    has_lengthscale = False

    def __init__(self, hidden_dim: int = 64, kernel_jitter: float = 1e-3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.covariance_network = CovarianceNetwork(hidden_dim=hidden_dim)
        self.raw_temporal_lengthscale = nn.Parameter(torch.tensor(0.0))
        self.kernel_jitter = kernel_jitter

    @property
    def temporal_lengthscale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_temporal_lengthscale) + 1e-3

    def _temporal_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        time_a = x1[:, :1]
        time_b = x2[:, :1]
        squared_distance = (time_a - time_b.transpose(0, 1)).pow(2)
        lengthscale = self.temporal_lengthscale
        return torch.exp(-0.5 * squared_distance / (lengthscale * lengthscale))

    def _spatial_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        spatial_a = x1[:, 1:]
        spatial_b = x2[:, 1:]
        sigma_a = self.covariance_network(spatial_a)
        sigma_b = self.covariance_network(spatial_b)

        sigma_sum = sigma_a[:, None, :, :] + sigma_b[None, :, :, :]
        delta = spatial_b[None, :, :] - spatial_a[:, None, :]
        eye = torch.eye(2, device=x1.device, dtype=x1.dtype).view(1, 1, 2, 2)
        inv_sigma = torch.linalg.inv(sigma_sum + self.kernel_jitter * eye)
        mahalanobis = torch.einsum("abc,abcd,abd->ab", delta, inv_sigma, delta)

        det_a = torch.linalg.det(sigma_a).clamp_min(1e-8)
        det_b = torch.linalg.det(sigma_b).clamp_min(1e-8)
        det_sum = torch.linalg.det(sigma_sum).clamp_min(1e-8)
        norm = (det_a.sqrt()[:, None] * det_b.sqrt()[None, :]).sqrt() / det_sum.sqrt()
        return norm * torch.exp(-0.5 * mahalanobis)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        covariance = self._temporal_kernel(x1, x2) * self._spatial_kernel(x1, x2)
        if x1.shape == x2.shape and torch.equal(x1, x2):
            covariance = covariance + self.kernel_jitter * torch.eye(
                x1.shape[0],
                device=x1.device,
                dtype=x1.dtype,
            )
        if diag:
            return covariance.diagonal(dim1=-2, dim2=-1)
        return covariance


class LatentKrillGP(ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, hidden_dim: int = 64, kernel_jitter: float = 1e-3) -> None:
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(DeepNonStationaryKernel(hidden_dim=hidden_dim, kernel_jitter=kernel_jitter))

    def forward(self, coords: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(coords)
        covar_x = self.covar_module(coords)
        return MultivariateNormal(mean_x, covar_x)


class ClimateConditionedGPModel(nn.Module):
    def __init__(self, num_features: int, inducing_points: torch.Tensor, hidden_dim: int = 64, kernel_jitter: float = 1e-3) -> None:
        super().__init__()
        self.mean_module = nn.Linear(num_features, 1)
        self.gp_model = LatentKrillGP(inducing_points=inducing_points, hidden_dim=hidden_dim, kernel_jitter=kernel_jitter)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.hotspot_scale = nn.Parameter(torch.tensor(1.0))
        self.hotspot_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor | MultivariateNormal]:
        latent_dist = self.gp_model(coords)
        latent_mean = latent_dist.mean.unsqueeze(-1)
        density = self.mean_module(features) + latent_mean
        hotspot_logits = self.hotspot_scale * latent_mean + self.hotspot_bias
        return {
            "latent_dist": latent_dist,
            "latent": latent_mean,
            "density": density,
            "hotspot_logits": hotspot_logits,
            "hotspot_probability": torch.sigmoid(hotspot_logits),
        }


class AutoregressiveGPModel(ClimateConditionedGPModel):
    pass


class LatentStateAugmentedGPModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        inducing_points: torch.Tensor,
        hidden_dim: int = 64,
        kernel_jitter: float = 1e-3,
        env_feature_count: int = 7,
        lag_feature_count: int = 3,
        latent_state_dim: int = 8,
    ) -> None:
        super().__init__()
        self.env_feature_count = env_feature_count
        self.lag_feature_count = lag_feature_count
        self.latent_state_dim = latent_state_dim
        self.gp_model = LatentKrillGP(inducing_points=inducing_points, hidden_dim=hidden_dim, kernel_jitter=kernel_jitter)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.hotspot_scale = nn.Parameter(torch.tensor(1.0))
        self.hotspot_bias = nn.Parameter(torch.tensor(0.0))

        if lag_feature_count > 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(lag_feature_count, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_state_dim),
                nn.Tanh(),
            )
            self.state_hotspot_head = nn.Linear(latent_state_dim, 1)
        else:
            self.state_encoder = None
            self.state_hotspot_head = None

        self.mean_module = nn.Linear(env_feature_count + latent_state_dim, 1)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor | MultivariateNormal]:
        env_features = features[:, : self.env_feature_count]
        if self.lag_feature_count > 0 and self.state_encoder is not None:
            lag_features = features[:, self.env_feature_count : self.env_feature_count + self.lag_feature_count]
            latent_state = self.state_encoder(lag_features)
            hotspot_state_term = self.state_hotspot_head(latent_state)
        else:
            latent_state = torch.zeros(features.shape[0], self.latent_state_dim, device=features.device, dtype=features.dtype)
            hotspot_state_term = 0.0

        latent_dist = self.gp_model(coords)
        latent_mean = latent_dist.mean.unsqueeze(-1)
        density = self.mean_module(torch.cat([env_features, latent_state], dim=1)) + latent_mean
        hotspot_logits = self.hotspot_scale * latent_mean + hotspot_state_term + self.hotspot_bias
        return {
            "latent_dist": latent_dist,
            "latent": latent_mean,
            "latent_state": latent_state,
            "density": density,
            "hotspot_logits": hotspot_logits,
            "hotspot_probability": torch.sigmoid(hotspot_logits),
        }


LagFeatureGPModel = AutoregressiveGPModel
FishCastModel = ClimateConditionedGPModel
