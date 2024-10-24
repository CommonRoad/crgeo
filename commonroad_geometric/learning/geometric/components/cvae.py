from typing import Optional, overload, Union, Callable, TypeVar, Generic, Tuple

import torch
from commonroad_geometric.common.torch_utils.helpers import assert_size
from torch import nn, Tensor
from torch.distributions import Normal

T_GeneratorOutput = TypeVar("T_GeneratorOutput")
T_Target = TypeVar("T_Target")

EPS = 1e-4


class ConditionalVariationalAutoencoder(nn.Module, Generic[T_GeneratorOutput, T_Target]):
    # z ~ p(z | x)
    # ŷ ~ p(ŷ | x, z)
    # x = input
    # z = latent
    # ŷ = output
    # y = target

    def __init__(
        self,
        recognition_network: nn.Module,
        prior_network: Optional[nn.Module],
        latent_dims: int,
        generator_network: nn.Module,
    ):
        super().__init__()
        self.latent_dims = latent_dims
        self.register_buffer("eps", torch.tensor([EPS], dtype=torch.float32), persistent=False)

        self.recognition_network = recognition_network  # recognition model q(z | x, y)
        self.generator_network = generator_network  # generation model p(y | x, z)
        self.prior_network = prior_network  # prior model p(z | x)
        # if prior_network is None we set p(z | x) = p(z) = N(z | 0, I) (standard multivariate normal)

        self.standard_normal_prior_distribution = Normal(
            loc=torch.zeros((latent_dims,)),
            scale=torch.ones((latent_dims,)),
        )

    @overload
    def forward(
        self,
        input: Tensor,
        target: T_GeneratorOutput,
    ) -> Tuple[T_GeneratorOutput, Tuple[Tensor, Tensor], T_GeneratorOutput, Optional[Tuple[Tensor, Tensor]]]:
        ...  # at train time

    @overload
    def forward(self, input: Tensor) -> T_GeneratorOutput:
        ...  # at test time

    def forward(
        self,
        input: Tensor,
        target: Optional[T_GeneratorOutput] = None,
    ) -> Union[Tuple[T_GeneratorOutput, Tuple[Tensor, Tensor], T_GeneratorOutput, Optional[Tuple[Tensor, Tensor]]], T_GeneratorOutput]:
        assert self.training == (target is not None)
        device = input.device
        N = input.size(0)

        if self.training:
            means, log_variances = self.recognition_network(input=input, target=target)
            assert_size(means, (N, self.latent_dims))
            assert_size(log_variances, (N, self.latent_dims))
            latent_distribution = Normal(loc=means, scale=torch.exp(log_variances) + self.eps)
            # rsample = reparameterized sample (applies reparameterization trick)
            latent_recognition = latent_distribution.rsample().to(device)
            assert_size(latent_recognition, (N, self.latent_dims))

            prior_params: Optional[Tuple[Tensor, Tensor]]
            if self.prior_network is None:
                prior_params = None
                latent_prior = self.standard_normal_prior_distribution.rsample(sample_shape=(N,)).to(device)
            else:
                prior_params = self.prior_network(input=input)
                means_prior, log_variances_prior = prior_params
                assert_size(means_prior, (N, self.latent_dims))
                assert_size(log_variances_prior, (N, self.latent_dims))
                latent_distribution_prior = Normal(loc=means_prior, scale=torch.exp(log_variances_prior) + self.eps)
                latent_prior = latent_distribution_prior.rsample().to(device)
            assert_size(latent_prior, (N, self.latent_dims))

            output_from_recognition = self.generator_network(input=input, latent=latent_recognition)
            assert output_from_recognition.size(0) == N
            output_from_prior = self.generator_network(input=input, latent=latent_prior)
            return output_from_recognition, (means, log_variances), output_from_prior, prior_params

        else:  # testing
            if self.prior_network is None:
                latent_prior = self.standard_normal_prior_distribution.rsample(sample_shape=(N,)).to(device)
            else:
                means_prior, log_variances_prior = self.prior_network(input=input)
                latent_distribution = Normal(loc=means_prior, scale=torch.exp(log_variances_prior) + self.eps)
                latent_prior = latent_distribution.rsample().to(device)

            output = self.generator_network(input=input, latent=latent_prior)
            return output

    def compute_cvae_loss(
        self,
        prediction: Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Optional[Tuple[Tensor, Tensor]]],
        target: T_GeneratorOutput,
        reconstruction_loss: Callable[..., Tensor],
    ) -> Tensor:
        output_from_recognition, recognition_params, output_from_prior, prior_params = prediction
        loss_reconstruction = reconstruction_loss(output=output_from_recognition, target=target)

        means_recognition, log_variances_recognition = recognition_params
        if prior_params is None:
            loss_kl = kl_divergence_standard_normal(means_recognition, log_variances_recognition)
        else:
            means_prior, log_variances_prior = prior_params
            loss_kl = kl_divergence_diagonal_normal(
                means_recognition, log_variances_recognition,
                means_prior, log_variances_prior,
            )

        loss_kl = loss_kl.mean()
        loss_cvae = loss_kl + loss_reconstruction
        return loss_cvae

    def compute_hybrid_loss(
        self,
        prediction: Tuple[T_GeneratorOutput, Tuple[Tensor, Tensor], T_GeneratorOutput, Optional[Tuple[Tensor, Tensor]]],
        target: T_Target,
        reconstruction_loss: Callable[..., Tensor],
        alpha: float,
    ) -> Tensor:
        assert 0.0 <= alpha <= 1.0
        _, _, output_from_prior, prior_params = prediction

        loss_cvae = self.compute_cvae_loss(
            prediction=prediction, target=target,
            reconstruction_loss=reconstruction_loss,
        )
        loss_gsnn = reconstruction_loss(output=output_from_prior, target=target)
        hybrid_loss = alpha * loss_cvae + (1. - alpha) * loss_gsnn
        return hybrid_loss


def kl_divergence_standard_normal(means: Tensor, log_variances: Tensor) -> Tensor:
    # compute the kl divergence between a multivariate normal with diagonal covariance matrix and a
    # standard multivariate normal of the same dimensionality
    return 0.5 * torch.sum(torch.exp(log_variances) + means**2 - log_variances - 1.0, dim=-1)


def kl_divergence_diagonal_normal(
    means1: Tensor, log_variances1: Tensor,
    means2: Tensor, log_variances2: Tensor,
) -> Tensor:
    variances1, variances2 = torch.exp(log_variances1) + EPS, torch.exp(log_variances2) + EPS
    return 0.5 * torch.sum(variances1 / variances2 + (means2 - means1)**2 /
                           variances2 + log_variances2 - log_variances1 - 1.0, dim=-1)
