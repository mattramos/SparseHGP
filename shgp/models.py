import tensorflow as tf
import numpy as np
import gpflow

from gpflow.config import default_jitter, default_float
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.mean_functions import Zero
from gpflow.models import BayesianModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

from copy import deepcopy
from tqdm import trange
import typing as tp


class SHGP(BayesianModel, InternalDataTrainingLossMixin):
    """Sparse hierarchical Gaussian Process.

    Inherits:
        BayesianModel. So we can use maximum log-likelihood as the training_loss
        InternalDataTrainingLossMixin. Because we require this model to hold it's own data
    """

    def __init__(
        self,
        data,
        inducing_points,
        group_kernel,
        individual_kernel,
        noise_variance=1.0,
        mean_function=None,
        
        name="SparseHierarchicalGP"
    ):
        """
        Args:
            data (Tuple): A tuple containing X and Y. X must be shape (n_data, n_covariates) and Y must be shape (n_data, n_time_series). X is descriptive of all Y aka, all samples in Y are collocated.
            inducing_points (array, optional): An array describing the initial inducing point locations.
            group_kernel (gpflow.kernels.Kernel): The kernel that describes the group level behaviour
            individual_kernel (gpflow.kernels.Kernel): The kernel that describes the individual level behaviour
            noise_variance (float, optional): The initial likelihood variance. Defaults to 1.0.
            mean_function (gpflow.mean_functions.MeanFunction, optional): The GP mean function. Defaults to a Zero mean function.
            
            name (str, optional): Name of model. Defaults to "SparseHierarchicalGP".
        """
        super().__init__()
        # Tensorise data
        X, Y = data
        self.X = data_input_to_tensor(X)  # n_data x n_covars
        self.Y = data_input_to_tensor(Y)  # n_data x n_reals
        tf.debugging.assert_shapes([(self.X, ("N", "C")), (self.Y, ("N", "K"))])

        # Inducing points
        self.inducing_points = inducingpoint_wrapper(inducing_points)

        # Check shapes
        tf.debugging.assert_shapes(
            [
                (self.X, ("N", "C")),
                (self.inducing_points.Z, ("M", "C")),
                (self.Y, ("N", "K"))
                ])

        # Useful constants
        self.num_inducing = inducing_points.shape[0]
        self.num_data = self.X.shape[0]
        self.num_individuals = self.Y.shape[1]

        # Copy kernels
        # Group kernel
        self.K_group = deepcopy(group_kernel)

        # Inidividual kernels
        self.K_individual_list = [deepcopy(individual_kernel) for i in range(self.num_individuals)]
        
        # Generate likelihood
        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(noise_variance)

        # Default is zero mean function
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function

    def maximum_log_likelihood_objective(self):
        """
        The objective for maximum likelihood estimation. In this case we maximise
        the lower bound to the log-marginal likelihood (ELBO).
        """
        return self.total_elbo()

    def total_elbo(self):
        """Calculates the ELBO for all the kernels.

        Returns:
            tf.Tensor: The value of the ELBO
        """

        # Initialise the ELBO at zero
        self.loss_sum = tf.constant(0.0, dtype=tf.float64)

        # Calculate elbo for all terms
        for i in range(len(self.K_individual_list)):
            for j in range(len(self.K_individual_list)):
                if i == j:
                    # If diagonal term add group and individual kernels
                    elbo_part = self.calc_ELBO_part(
                        self.X,
                        tf.expand_dims(self.Y[:, i], -1),
                        kernel=self.K_individual_list[i] + self.K_group,
                    )
                else:
                    # If off diagonal term use group kernel
                    elbo_part = self.calc_ELBO_part(
                        self.X, tf.expand_dims(self.Y[:, i], -1), kernel=self.K_group
                    )
                self.loss_sum = tf.add(self.loss_sum, elbo_part)

        # We return the total elbo (to be maximised)
        return self.loss_sum 

    def calc_ELBO_part(self, X, Y, kernel):
        """Calculates the ELBO given a kernel

        Args:
            X (tf.Tensor): The X data (covaraites)
            Y (tf.Tensor): The Y data (data to be fit)
            kernel (gpflow.kernels.Kernel): The GP kernel

        Returns:
            tf.Tensor: The ELBO component for the given kernel
        """
        noise = self.likelihood.variance
        n = X.shape[0]
        z = self.inducing_points.Z
        Kzz = kernel.K(z, z)
        Kzz += tf.eye(self.num_inducing, dtype=tf.float64) * default_jitter()
        Kzx = kernel.K(z, X)
        Kxx_diag = kernel(X, full_cov=False)
        mux = self.mean_function(X)

        Lz = tf.linalg.cholesky(Kzz)

        A = tf.linalg.triangular_solve(Lz, Kzx, lower=True) / tf.math.sqrt(noise)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)

        B = tf.eye(self.num_inducing, dtype=tf.float64) + AAT

        L = tf.linalg.cholesky(B)

        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        diff = Y - mux

        L_inv_A_diff = tf.linalg.triangular_solve(L, tf.linalg.matmul(A, diff), lower=True)

        quad = (tf.reduce_sum(tf.math.square(diff)) - tf.reduce_sum(tf.math.square(L_inv_A_diff))) / noise

        two_log_prob = - n * tf.math.log(2.0 * noise) - log_det_B - quad

        two_trace = tf.reduce_sum(Kxx_diag) / noise - tf.reduce_sum(tf.linalg.diag_part(AAT))

        return (two_log_prob - two_trace) / 2.0


    def fit(self, params, compile: bool = False):
        """Fit the HSGP using an Adam optimiser.

        Args:
            params (dict): A dictionary containing the fitting parameters: 
                learning_rate - the Adam learning rate,
                optim_nits - the number of optimisation steps,

            compile (bool, optional): Whether to compile the function for speed. Defaults to False.
        """
        self.objective_evals = []
        opt = tf.optimizers.Adam(learning_rate=params["learning_rate"])

        # Comiple function for speed
        if compile:
            objective = tf.function(self.training_loss)
        else:
            objective = self.training_loss

        tr = trange(params["optim_nits"])
        for i in tr:
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                loss = objective()
                self.objective_evals.append(loss.numpy())
            grads = tape.gradient(loss, self.trainable_variables)
            opt.apply_gradients(zip(grads, self.trainable_variables))
            if i % params['log_interval'] == 0:
                tr.set_postfix(loss=loss.numpy())

    def predict_f(
        self,
        xtest: tp.Union[np.ndarray, tf.Tensor],
        kernel: gpflow.kernels.base.Kernel,
        Y: tp.Union[np.ndarray, tf.Tensor],
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        """Compute the mean and variance of the latent function at new points.

        Args:
            xtest (tp.Union[np.ndarray, tf.Tensor]): The new test points at which to predict
            kernel (gpflow.kernels.base.Kernel): The kernel to use
            Y (tp.Union[np.ndarray, tf.Tensor]): The Y vector related to the kernel

        Returns:
            tp.Tuple[tf.Tensor, tf.Tensor]: Latent function mean and variance
        """
        Xi = self.X
        Yi = tf.expand_dims(Y, axis=-1)
        # For group kernels average the Ys. Where Y is just (D x 1),
        # the output will still be the same
        diff = Yi - self.mean_function(Xi)

        kuf = Kuf(self.inducing_points, kernel, Xi)
        kuu = Kuu(self.inducing_points, kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_points, kernel, xtest)

        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(self.num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, diff)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        f_mean = mean + self.mean_function(xtest)
        var = (
            kernel(xtest, full_cov=False)
            + tf.reduce_sum(tf.square(tmp2), 0)
            - tf.reduce_sum(tf.square(tmp1), 0)
        )
        f_var = tf.tile(var[:, None], [1, 1])

        return f_mean, f_var

    def predict_group(
        self, xtest: tp.Union[np.ndarray, tf.Tensor]
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        """Predict the group level mean and variance

        Args:
            xtest (tp.Union[np.ndarray, tf.Tensor]): The new test points at which to predict

        Returns:
            tp.Tuple[tf.Tensor, tf.Tensor]: Mean and variance of the group function
        """
        # TODO - is this averaging right?
        f_mean, f_var = self.predict_f(xtest, self.K_group, tf.reduce_mean(self.Y, axis=1))
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_individual(
        self, xtest: tp.Union[np.ndarray, tf.Tensor], individual_idx: int
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        """Predict the individual (including group function) mean and variance

        Args:
            xtest (tp.Union[np.ndarray, tf.Tensor]): The new test points at which to predict
            individual_idx (int): _description_

        Returns:
            tp.Tuple[tf.Tensor, tf.Tensor]: Mean and variance of the individual 
        """
        f_mean, f_var = self.predict_f(
            xtest, self.K_group + self.K_individual_list[individual_idx], self.Y[:, individual_idx]
        )
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    # def predict_individual_without_group(
    #     self, xtest: tp.Union[np.ndarray, tf.Tensor], individual_idx: int
    # ) -> tp.Tuple[tf.Tensor, tf.Tensor]:

    #     f_mean, f_var = self.predict_f(
    #         xtest, self.K_individual_list[individual_idx], self.Y[:, individual_idx]
    #     )
    #     # return f_mean, f_var
    #     return self.likelihood.predict_mean_and_var(f_mean, f_var)