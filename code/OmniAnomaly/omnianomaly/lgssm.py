import collections
from typing import Optional, Tuple, List

import torch as t
import torch.nn as nn
from torch.distributions import MultivariateNormal, Distribution
from data_config import *


class LinearGaussianStateSpaceModel:

    def __init__(
            self,
            transition_matrix: t.Tensor,
            observation_matrix: t.Tensor,
            transition_noise: MultivariateNormal,
            observation_noise: MultivariateNormal,
            init_state_prior: MultivariateNormal,
            num_timesteps: int,
    ):
        """
        A Linear Gaussian State Space Model implementing the following generative process

        p(z_0) = N(z_0 | prior_mean, prior_covariance)
        p(z_t|z_{t-1}}) = N(z_t |  z_{t-1} @ transition_matrix, transition_covariance)
        p(x_t|z_t) = N(x_t | z_t @ observation_matrix, observation_covariance)

        :param prior_mean: (z_size, )
        :param prior_covariance: (z_size, z_size)

        :param transition_matrix: (z_size, z_size)
        :param transition_covariance: (z_size, z_size)

        :param observation_matrix: (z_size, x_size)
        :param observation_covariance: (x_size, x_size)
        """
        super().__init__()
        self.transition_matrix = nn.Parameter(transition_matrix, requires_grad=True)
        self.observation_matrix = nn.Parameter(observation_matrix, requires_grad=True)
        self.transition_noise = transition_noise
        self.observation_noise = observation_noise
        self.init_state_prior = init_state_prior
        self.num_timesteps = num_timesteps
        self.z_dim, _ = transition_matrix.size()

    def sample(self, sample_n_shape: tuple):
        # with torch.no_grad():
        latent_previous = self.init_state_prior.sample(sample_shape=sample_n_shape).unsqueeze(-1)
        # print(f"latent_previous:{latent_previous.shape}")
        observation_res = torch.zeros((self.num_timesteps,)+latent_previous.size())
        for i in range(self.num_timesteps):
            # print(f"self.transition_matrix:{self.transition_matrix.shape}")
            latent_predict = self.transition_matrix.matmul(latent_previous)
            latent_sampled = latent_predict + self.transition_noise.sample(sample_shape=sample_n_shape).unsqueeze(-1)
            observation_predict = self.observation_matrix.matmul(latent_sampled)
            observation_sampled = observation_predict + self.observation_noise.sample(sample_shape=sample_n_shape).unsqueeze(-1)
            # print(f"i:{i} observation_sampled:{observation_sampled}")
            observation_res[i] = observation_sampled
            latent_previous = latent_sampled
        return observation_res
    
    def _propagate_mean(self, p_mean: t.Tensor, other_mat: t.Tensor, distribution: MultivariateNormal):
        return other_mat.matmul(p_mean) + distribution.mean.unsqueeze(dim=-1)
    
    def _propagate_cov(self, p_cov: t.Tensor, other_mat: t.Tensor, distribution: MultivariateNormal):
        return other_mat.matmul(p_cov).matmul(other_mat.transpose(-1, -2)) + distribution.covariance_matrix
    
    def log_prob(self, x_observed: t.Tensor):
        """Run a Kalman filter over a provided sequence of outputs.
            Note` that the returned values `filtered_means`, `predicted_means`, and
            `observation_means` depend on the observed time series `x`, while the
            corresponding covariances are independent of the observed series; i.e., they
            depend only on` the model itself. 
        """
        """
            Returns:
            log_likelihoods: Per-timestep log marginal likelihoods `log
                p(x_t | x_{:t-1})` evaluated at the input `x`, as a `Tensor`
                of shape `sample_shape(x) + batch_shape + [num_timesteps].`
            filtered_means: Means of the per-timestep filtered marginal
                distributions p(z_t | x_{:t}), as a Tensor of shape
                `sample_shape(x) + batch_shape + [num_timesteps, latent_size]`.
            filtered_covs: Covariances of the per-timestep filtered marginal
                distributions p(z_t | x_{:t}), as a Tensor of shape
                `sample_shape(mask) + batch_shape + [num_timesteps, latent_size,
                latent_size]`. Note that the covariances depend only on the model and
                the mask, not on the data, so this may have fewer dimensions than
                `filtered_means`.
            predicted_means: Means of the per-timestep predictive
                distributions over latent states, p(z_{t+1} | x_{:t}), as a
                Tensor of shape `sample_shape(x) + batch_shape +
                [num_timesteps, latent_size]`.
            predicted_covs: Covariances of the per-timestep predictive
                distributions over latent states, p(z_{t+1} | x_{:t}), as a
                Tensor of shape `sample_shape(mask) + batch_shape +
                [num_timesteps, latent_size, latent_size]`. Note that the covariances
                depend only on the model and the mask, not on the data, so this may
                have fewer dimensions than `predicted_means`.
            observation_means: Means of the per-timestep predictive
                distributions over observations, p(x_{t} | x_{:t-1}), as a
                Tensor of shape `sample_shape(x) + batch_shape +
                [num_timesteps, observation_size]`.
            observation_covs: Covariances of the per-timestep predictive
                distributions over observations, p(x_{t} | x_{:t-1}), as a
                Tensor of shape `sample_shape(mask) + batch_shape + [num_timesteps,
                observation_size, observation_size]`. Note that the covariances depend
                only on the model and the mask, not on the data, so this may have fewer
                dimensions than `observation_means`.
        """

        x_observed = x_observed.permute(2, 0, 1, 3)

        x_observed = x_observed.unsqueeze(dim=-1)
        # Initialize filtering distribution from the prior. The mean in
        # a Kalman filter depends on data, so should match the full
        # sample and batch shape. The covariance is data-independent, so
        # only has batch shape.
        prior_mean = self.init_state_prior.mean.unsqueeze(dim=-1).broadcast_to(x_observed.size()[1:-2]+(3, 1))
        prior_cov = self.init_state_prior.covariance_matrix.broadcast_to(x_observed.size()[1:-2]+(3, 3))
        initial_observation_mean = self._propagate_mean(prior_mean, self.transition_matrix, self.observation_noise)
        initial_observation_cov = self._propagate_cov(prior_cov, self.transition_matrix, self.observation_noise)
        initial_state = KalmanFilterState(
          predicted_mean=prior_mean,
          predicted_cov=prior_cov,
          filtered_mean=prior_mean,  # establishes shape, value ignored
          filtered_cov=prior_cov,  # establishes shape, value ignored
          observation_mean=initial_observation_mean,
          observation_cov=initial_observation_cov,
        #   log_marginal_likelihood=t.zeros(size=prior_mean.size())
          )
        log_marginal_likelihood = t.zeros(size=(self.num_timesteps,)+prior_mean.size()[:-2]).to(device=global_device)
        state = initial_state
        # with torch.no_grad():
        for time_step in range(self.num_timesteps):
            # print(f"time_step:{time_step}")
            filtered_mean, filtered_cov, observation_dist = self.linear_gaussian_update(
                state.predicted_mean, 
                state.predicted_cov,
                self.observation_matrix, 
                self.observation_noise,
                x_observed[time_step]
            )
            # Compute the marginal likelihood p(x_{t} | x_{:t-1}) for this
            # observation.
            log_marginal_likelihood[time_step] = observation_dist.log_prob(x_observed[time_step][..., 0])
            
            # Run the filtered posterior through the transition
            # model to predict the next time step:
            #  u_{t|t-1} = F_t u_{t-1} + b_t
            #  P_{t|t-1} = F_t P_{t-1} F_t' + Q_t
            predicted_mean, predicted_cov = self.kalman_transition(
                filtered_mean,
                filtered_cov)
            
            state = KalmanFilterState(
                filtered_mean, filtered_cov,
                predicted_mean, predicted_cov,
                observation_dist.mean.unsqueeze(-1),
                observation_dist.covariance_matrix)
                
        # print(f"log_marginal_likelihood.size():{log_marginal_likelihood.size()}")
        return log_marginal_likelihood.permute(1, 2, 0)
            

    def kalman_transition(self, filtered_mean, filtered_cov):
        """Propagate a filtered distribution through a transition model."""

        predicted_mean = self._propagate_mean(filtered_mean,
                                        self.transition_matrix,
                                        self.transition_noise)
        predicted_cov = self._propagate_cov(filtered_cov,
                                        self.transition_matrix,
                                        self.transition_noise)
        return predicted_mean, predicted_cov

    def linear_gaussian_update(self, prior_mean: t.Tensor, prior_cov: t.Tensor, observation_matrix: t.Tensor, observation_noise: t.Tensor, x_observed: t.Tensor):
    # -> tuple(t.Tensor, t.Tensor, MultivariateNormal):
        """Conjugate update for a linear Gaussian model.

        Given a normal prior on a latent variable `z`,
            `p(z) = N(prior_mean, prior_cov) = N(u, P)`,
        for which we observe a linear Gaussian transformation `x`,
            `p(x|z) = N(H * z + c, R)`,
        the posterior is also normal:
            `p(z|x) = N(u*, P*)`.

        We can write this update as
            x_expected = H * u + c # pushforward prior mean
            S = R + H * P * H'  # pushforward prior cov
            K = P * H' * S^{-1} # optimal Kalman gain
            u* = u + K * (x_observed - x_expected) # posterior mean
            P* = (I - K * H) * P (I - K * H)' + K * R * K' # posterior cov
        (see, e.g., https://en.wikipedia.org/wiki/Kalman_filter#Update)

        Args:
            prior_mean: `Tensor` with event shape `[latent_size, 1]` and
            potential batch shape `B = [b1, ..., b_n]`.
            prior_cov: `Tensor` with event shape `[latent_size, latent_size]`
            and batch shape `B` (matching `prior_mean`).
            observation_matrix: `LinearOperator` with shape
            `[observation_size, latent_size]` and batch shape broadcastable
            to `B`.
            observation_noise: potentially-batched
            `MultivariateNormalLinearOperator` instance with event shape
            `[observation_size]` and batch shape broadcastable to `B`.
            x_observed: potentially batched `Tensor` with event shape
            `[observation_size, 1]` and batch shape `B`.

        Returns:
            posterior_mean: `Tensor` with event shape `[latent_size, 1]` and
            batch shape `B`.
            posterior_cov: `Tensor` with event shape `[latent_size,
            latent_size]` and batch shape `B`.
            predictive_dist: the prior predictive distribution `p(x|z)`,
            as a `Distribution` instance with event
            shape `[observation_size]` and batch shape `B`. This will
            typically be `tfd.MultivariateNormalTriL`, but when
            `observation_size=1` we return a `tfd.Independent(tfd.Normal)`
            instance as an optimization.
        """
        # Push the predicted mean for the latent state through the
        # observation model
        x_expected = self._propagate_mean(prior_mean, observation_matrix, observation_noise)
        # print(f"x_expected:{x_expected.size()} {x_expected}")
        # print(f"x_observed:{x_observed.size()} {x_observed}")
        # Push the predictive covariance of the latent state through the
        # observation model:
        #  S = R + H * P * H'.
        # We use a temporary variable for H * P,
        # reused below to compute Kalman gain.
        tmp_obs_cov = observation_matrix.matmul(prior_cov) # H*P
        # print(f"H*P:{tmp_obs_cov.size()} {tmp_obs_cov}")

        predicted_obs_cov = (
            observation_matrix.matmul(tmp_obs_cov)
            .matmul(observation_matrix.transpose(-1, -2)) 
            + observation_noise.covariance_matrix
            ) # H*P*H.T +R
        # print(f"H*P*H.T +R:{predicted_obs_cov}")
        # Compute optimal Kalman gain:
        #  K = P * H' * S^{-1}
        # Since both S and P are cov matrices, thus symmetric,
        # we can take the transpose and reuse our previous
        # computation:
        #      = (S^{-1} * H * P)'
        #      = (S^{-1} * tmp_obs_cov) '
        #      = (S \ tmp_obs_cov)'
        predicted_obs_cov_chol = t.linalg.cholesky(predicted_obs_cov)
        # print(f"L:{predicted_obs_cov_chol}")
        gain_transpose = t.cholesky_solve(predicted_obs_cov_chol, tmp_obs_cov)
        # print(f"K.T:{gain_transpose}")
        gain = gain_transpose.transpose(-1, -2)
        
        # Compute the posterior mean, incorporating the observation.
        #  u* = u + K (x_observed - x_expected)
        posterior_mean = (prior_mean + t.matmul(gain, x_observed - x_expected))
        # print(f"posterior_mean:{posterior_mean}")
        
        # For the posterior covariance, we could use the simple update
        #  P* = P - K * H * P
        # but this is prone to numerical issues because it subtracts a
        # value from a PSD matrix.  We choose instead to use the more
        # expensive Jordan form update
        #  P* = (I - K H) * P * (I - K H)' + K R K'
        # which always produces a PSD result. This uses
        #  tmp_term = (I - K * H)'
        # as an intermediate quantity.
        tmp_term = -gain.matmul(observation_matrix)  # -K * H
        # print(f"before add: {tmp_term}")
        # tmp_term = t.diagonal_scatter(tmp_term, t.diagonal(tmp_term) + 1) 
        tmp_term = tmp_term + t.eye(self.z_dim).to(global_device) 
        # print(f"after add: {tmp_term}")
        
        posterior_cov = tmp_term.matmul(prior_cov).matmul(tmp_term.transpose(-1, -2)) + gain.matmul(self.observation_noise.covariance_matrix).matmul(gain_transpose)
        # print(f"posterior_cov:{posterior_cov}")
        
        predictive_dist = MultivariateNormal(
        loc=x_expected[..., 0],
        scale_tril=predicted_obs_cov_chol)
        return posterior_mean, posterior_cov, predictive_dist


KalmanFilterState = collections.namedtuple("KalmanFilterState", [
    "filtered_mean", "filtered_cov",
    "predicted_mean", "predicted_cov",
    "observation_mean", "observation_cov"])
            
