import pytest
import numpy as np
from shgp import SHGP
import gpflow
import tensorflow as tf

def make_data(n_time_series=3, n_data=10, n_dims=1):
    X = np.random.randn(n_data, n_dims)
    Xtest = np.random.randn(n_data, n_dims)
    y = np.random.randn(n_data, n_time_series)
    return X, Xtest, y


@pytest.mark.parametrize("n_data", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("n_inducing", [1, 5, 12])
@pytest.mark.parametrize("n_time_series", [1, 2, 5])
def test_SHGP(n_data, n_dims, n_inducing, n_time_series):
    X, Xtest, y = make_data(n_time_series, n_data, n_dims)

    data = (X, y)
    Z = np.linspace(np.asarray([-3.] * n_dims), np.asarray([3] * n_dims), n_inducing)

    kernel = gpflow.kernels.Matern32()

    shgp = SHGP(
        data,
        inducing_points=Z,
        group_kernel=kernel,
        individual_kernel=kernel
        )

    params = dict()
    params['optim_nits'] = 20
    params['log_interval']= 1
    params['learning_rate'] = 0.05

    shgp.fit(params, compile=False)

    for indi_idx in range(n_time_series):
        mu, sigma = shgp.predict_individual(Xtest, indi_idx)
        assert isinstance(mu, tf.Tensor)
        assert isinstance(sigma, tf.Tensor)
        assert mu.numpy().shape == (n_data, 1)
        assert sigma.numpy().shape == (n_data, 1)

    mu, sigma = shgp.predict_group(Xtest)
    assert isinstance(mu, tf.Tensor)
    assert isinstance(sigma, tf.Tensor)
    assert mu.numpy().shape == (n_data, 1)
    assert sigma.numpy().shape == (n_data, 1)
