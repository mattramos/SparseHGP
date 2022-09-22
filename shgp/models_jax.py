from re import S
import numpy as np
from copy import deepcopy

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit
import gpjax as gpx
from gpjax.parameters import transform
from gpjax.kernels import cross_covariance, diagonal, gram
from gpjax.utils import I, concat_dictionaries
from jax import vmap
import jax.scipy as jsp
import jax.lax as lax

class SHGP_jax:

    def __init__(
        self,
        data,
        inducing_points,
        group_kernel,
        individual_kernel,
        likelihood,
        noise_variance=1.0,
        mean_function=gpx.mean_functions.Zero(),
        key=jr.PRNGKey(123),
        name="SparseHierarchicalGPinJAX"
    ):

        # Data
        self.x = jnp.asarray(data[0])
        self.y = jnp.asarray(data[1])
        assert self.x.shape[0] == self.y.shape[0], "The number of data points must be the same for both x and y"
        self.num_individuals = jnp.asarray(self.y.shape[1])
        self.num_datapoints = jnp.asarray(self.x.shape[0])

        # Inducing points need to be trainable
        self.inducing_points = jnp.asarray(inducing_points)
        self.num_inducing = jnp.asarray(inducing_points.shape[0])

        # Kernels
        self.K_group = deepcopy(group_kernel)
        self.K_individual = individual_kernel
        self.kernels = []
        for i in range(self.num_individuals):
            for j in range(self.num_individuals):
                if i == j:
                    kernel = self.K_individual + self.K_group
                    self.kernels.append(kernel)
                else:
                    kernel = self.K_group
                    self.kernels.append(kernel)

        # Likelihood
        self.likelihood = likelihood

        # Mean function
        self.mean_function = mean_function 

        # Other parameters
        self.default_jitter = jnp.asarray(1.e-6)

        # Initialise the parameters of the model
        # Collect all parameters, trainables, constrainers and unconstrainers
        self.parameter_state_dict = dict()
        parameters = dict()
        trainables = dict()
        constrainers = dict()
        unconstrainers = dict()

        # Inducing points
        parameters['inducing_points'] = dict(inducing_points=self.inducing_points)
        trainables['inducing_points'] = dict(inducing_points=True)
        constrainers['inducing_points'] = dict(inducing_points=jnp.asarray) # TODO this is totally hacky and horrible...
        unconstrainers['inducing_points'] = dict(inducing_points=jnp.asarray) # Could use linear bijector?

        # Kernels
        for i in range(self.num_individuals):
            name = f'kernel{i}'
            k_param_state = gpx.initialise(self.K_individual, key)
            parameters[name] = k_param_state.params
            trainables[name] = k_param_state.trainables
            constrainers[name] = k_param_state.constrainers
            unconstrainers[name] = k_param_state.unconstrainers
        group_name = 'kernel_group'
        kg_param_state = gpx.initialise(self.K_group, key)
        parameters[group_name] = kg_param_state.params
        trainables[group_name] = kg_param_state.trainables
        constrainers[group_name] = kg_param_state.constrainers
        unconstrainers[group_name] = kg_param_state.unconstrainers

        # # Kernels
        # kernels = []
        # for i in range(len(self.K_individual_list)):
        #     for j in range(len(self.K_individual_list)):
        #         name = f'kernel{i}'
        #         if i == j:
        #             kernel = self.K_individual_list[i] + self.K_group
        #         else:
        #             kernel = self.K_group
                




        #     k_param_state = gpx.initialise(k, key)
        #     parameters[name] = k_param_state.params
        #     trainables[name] = k_param_state.trainables
        #     constrainers[name] = k_param_state.constrainers
        #     unconstrainers[name] = k_param_state.unconstrainers
        # group_name = 'kernel_group'
        # kg_param_state = gpx.initialise(self.K_group, key)
        # parameters[group_name] = kg_param_state.params
        # trainables[group_name] = kg_param_state.trainables
        # constrainers[group_name] = kg_param_state.constrainers
        # unconstrainers[group_name] = kg_param_state.unconstrainers

        # Likelihood - currently not initialising the noise variance
        like_name = 'likelihood'
        likelihood_parameter_states = gpx.initialise(self.likelihood, key)
        parameters[like_name] = likelihood_parameter_states.params
        trainables[like_name] = likelihood_parameter_states.trainables
        constrainers[like_name] = likelihood_parameter_states.constrainers
        unconstrainers[like_name] = likelihood_parameter_states.unconstrainers

        # Mean function
        mf_name = 'mean_function'
        mf_param_state = gpx.initialise(self.mean_function, key)
        parameters[mf_name] = mf_param_state.params
        trainables[mf_name] = mf_param_state.trainables
        constrainers[mf_name] = mf_param_state.constrainers
        unconstrainers[mf_name] = mf_param_state.unconstrainers

        self.parameter_state_dict['params'] = parameters
        self.parameter_state_dict['trainables'] = trainables
        self.parameter_state_dict['constrainers'] = constrainers
        self.parameter_state_dict['unconstrainers'] = unconstrainers


    def total_elbo(self, params):

        # Need to calculate the elbo for each individual and then sum them up
        # This will be a function of the parameters of the model

        # Calculate the elbo for each individual
        # TODO: can vmap\

        loss_sum = 0.0
        for i in jnp.arange(self.num_individuals):
            for j in jnp.arange(self.num_individuals):
                if i == j:
                    kernel = self.K_individual + self.K_group
                    kernel_params = [params[f'kernel{i}'], params['kernel_group']]
                    kernel_unconstrainers = [self.parameter_state_dict['unconstrainers'][f'kernel{i}'], self.parameter_state_dict['unconstrainers']['kernel_group']]
                else:
                    kernel = self.K_group
                    kernel_params = params['kernel_group']
                    kernel_unconstrainers = self.parameter_state_dict['unconstrainers']['kernel_group']

                y = self.y[:, i]
                loss_sum = loss_sum + self.calc_elbo_part(kernel, y, kernel_params, params, kernel_unconstrainers)

        # loss_sum = lax.fori_loop(0, 2, lambda i, loss_sum: self.calc_elbo_part(i, j, params), 0)

        return loss_sum

    # @jit
    def calc_elbo_part(self, kernel, y, kernel_params, params, kernel_unconstrainers):
        # Straight lift from GPJax
        m = self.num_inducing
        x, n = self.x, self.num_datapoints

        unconstrainers = self.parameter_state_dict['unconstrainers']
        params = transform(params, unconstrainers)
        kernel_params = transform(kernel_params, kernel_unconstrainers)

        noise = params['likelihood']['obs_noise']
        z = params['inducing_points']['inducing_points']

        Kzz = gram(kernel, z, kernel_params)
        Kzz += I(m) * self.default_jitter
        Kzx = cross_covariance(kernel, z, x, kernel_params)
        Kxx_diag = vmap(kernel, in_axes=(0, 0, None))(
            x, x, kernel_params
        )

        μx = self.mean_function(x, params['mean_function'])

        Lz = jnp.linalg.cholesky(Kzz)

        # Notation and derivation:
        #
        # Let Q = KxzKzz⁻¹Kzx, we must compute the log normal pdf:
        #
        #   log N(y; μx, σ²I + Q) = -nπ - n/2 log|σ²I + Q| - 1/2 (y - μx)ᵀ (σ²I + Q)⁻¹ (y - μx).
        #
        # The log determinant |σ²I + Q| is computed via applying the matrix determinant lemma
        #
        #   |σ²I + Q| = log|σ²I| + log|I + Lz⁻¹ Kzx (σ²I)⁻¹ Kxz Lz⁻¹| = log(σ²) +  log|B|,
        #
        #   with B = I + AAᵀ and A = Lz⁻¹ Kzx / σ.
        #
        # Similary we apply matrix inversion lemma to invert σ²I + Q
        #
        #   (σ²I + Q)⁻¹ = (Iσ²)⁻¹ - (Iσ²)⁻¹ Kxz Lz⁻ᵀ (I + Lz⁻¹ Kzx (Iσ²)⁻¹ Kxz Lz⁻ᵀ )⁻¹ Lz⁻¹ Kzx (Iσ²)⁻¹
        #               = (Iσ²)⁻¹ - (Iσ²)⁻¹ σAᵀ (I + σA (Iσ²)⁻¹ σAᵀ)⁻¹ σA (Iσ²)⁻¹
        #               = I/σ² - Aᵀ B⁻¹ A/σ²,
        #
        # giving the quadratic term as
        #
        #   (y - μx)ᵀ (σ²I + Q)⁻¹ (y - μx) = [(y - μx)ᵀ(y - µx)  - (y - μx)ᵀ Aᵀ B⁻¹ A (y - μx)]/σ²,
        #
        #   with A and B defined as above.

        A = jsp.linalg.solve_triangular(Lz, Kzx, lower=True) / jnp.sqrt(noise)

        # AAᵀ
        AAT = jnp.matmul(A, A.T)

        # B = I + AAᵀ
        B = I(m) + AAT

        # LLᵀ = I + AAᵀ
        L = jnp.linalg.cholesky(B)

        # log|B| = 2 trace(log|L|) = 2 Σᵢ log Lᵢᵢ  [since |B| = |LLᵀ| = |L|²  => log|B| = 2 log|L|, and |L| = Πᵢ Lᵢᵢ]
        log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))

        diff = y - μx

        # L⁻¹ A (y - μx)
        L_inv_A_diff = jsp.linalg.solve_triangular(
            L, jnp.matmul(A, diff), lower=True
        )

        # (y - μx)ᵀ (Iσ² + Q)⁻¹ (y - μx)
        quad = (jnp.sum(diff**2) - jnp.sum(L_inv_A_diff**2)) / noise

        # 2 * log N(y; μx, Iσ² + Q)
        two_log_prob = -n * jnp.log(2.0 * jnp.pi * noise) - log_det_B - quad

        # 1/σ² tr(Kxx - Q) [Trace law tr(AB) = tr(BA) => tr(KxzKzz⁻¹Kzx) = tr(KxzLz⁻ᵀLz⁻¹Kzx) = tr(Lz⁻¹Kzx KxzLz⁻ᵀ) = trace(σ²AAᵀ)]
        two_trace = jnp.sum(Kxx_diag) / noise - jnp.trace(AAT)

        # log N(y; μx, Iσ² + KxzKzz⁻¹Kzx) - 1/2σ² tr(Kxx - KxzKzz⁻¹Kzx)
        return -1. * (two_log_prob - two_trace).squeeze() / 2.0

    def fit(self, n_iters=100, learning_rate=0.01):
        # loss = jit(self.total_elbo)
        params = self.parameter_state_dict['params']
        trainables = self.parameter_state_dict['trainables']
        loss = self.total_elbo
        opt = ox.adam(learning_rate=learning_rate)
        inference_state = gpx.fit(
            loss,
            params,
            trainables,
            opt,
            n_iters=n_iters,
        )

        return inference_state.unpack()

    def predict(self, X_new):
        pass

    def predict_f(self, X_new):
        pass