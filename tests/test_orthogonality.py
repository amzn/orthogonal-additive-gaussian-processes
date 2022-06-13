# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import pytest
# -

from oak.input_measures import (
    EmpiricalMeasure,
    GaussianMeasure,
    MOGMeasure,
    UniformMeasure,
)
from oak.model_utils import estimate_one_dim_gmm
from oak.ortho_rbf_kernel import OrthogonalRBFKernel

_THRESHOLD_NUMERICAL_ACCURACY = 2
np.random.seed(0)


@pytest.mark.parametrize(
    "kernel",
    [gpflow.kernels.RBF(lengthscales=10)],
)
def test_cov_X_s_Gaussian(kernel: gpflow.kernels.Kernel):
    k = OrthogonalRBFKernel(kernel, GaussianMeasure(0, 1))
    k_cov = k.cov_X_s(np.zeros((1, 1)))

    samples_std_normal = np.random.normal(size=10000)[:, None]
    k_cov_numeric = np.mean(k.base_kernel.K(np.zeros((1, 1)), samples_std_normal))
    print(np.abs(k_cov - k_cov_numeric))
    np.testing.assert_almost_equal(
        np.abs(k_cov - k_cov_numeric), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY
    )


@pytest.mark.parametrize(
    "kernel",
    [gpflow.kernels.RBF(lengthscales=10)],
)
def test_var_s_Gaussian(kernel: gpflow.kernels.Kernel):
    k = OrthogonalRBFKernel(kernel, GaussianMeasure(0, 1))
    k_var = k.var_s()

    samples_std_normal = np.random.normal(size=10000)[:, None]
    k_var_numeric = np.mean(k.cov_X_s(samples_std_normal))
    print(np.abs(k_var - k_var_numeric))
    np.testing.assert_almost_equal(
        np.abs(k_var - k_var_numeric), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY
    )


def test_cov_X_s_Uniform():
    k = OrthogonalRBFKernel(gpflow.kernels.RBF(lengthscales=10), UniformMeasure(0, 1))
    k_cov = k.cov_X_s(np.zeros((1, 1)))

    samples_std_normal = np.random.uniform(size=10000)[:, None]
    k_cov_numeric = np.mean(k.base_kernel.K(np.zeros((1, 1)), samples_std_normal))
    print(np.abs(k_cov - k_cov_numeric))
    np.testing.assert_almost_equal(
        np.abs(k_cov - k_cov_numeric), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY
    )


def test_var_s_Uniform():
    k = OrthogonalRBFKernel(gpflow.kernels.RBF(lengthscales=10), UniformMeasure(0, 1))
    k_var = k.var_s()

    samples_std_normal = np.random.uniform(size=10000)[:, None]
    k_var_numeric = np.mean(k.cov_X_s(samples_std_normal))
    print(np.abs(k_var - k_var_numeric))
    np.testing.assert_almost_equal(
        np.abs(k_var - k_var_numeric), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY
    )


@pytest.mark.parametrize(
    "kernel",
    [gpflow.kernels.RBF(lengthscales=10)],
)
def test_GaussianMeasure(kernel: gpflow.kernels.Kernel):
    N = 1000
    k = OrthogonalRBFKernel(kernel, GaussianMeasure(0, 1))
    xx = np.random.normal(0, 1, (N, 1))
    mu = np.zeros(N)
    f = np.random.multivariate_normal(mu, k.K(xx), size=1)
    np.testing.assert_almost_equal(f.mean(), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY)


def test_UniformMeasure():
    N = 1000
    k = OrthogonalRBFKernel(gpflow.kernels.RBF(lengthscales=10), UniformMeasure(0, 1))
    xx = np.random.uniform(0, 1, (N, 1))
    mu = np.zeros(N)
    f = np.random.multivariate_normal(mu, k.K(xx), size=1)
    np.testing.assert_almost_equal(f.mean(), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY)


def test_EmpiricalMeasure():
    N = 1000
    location = np.reshape(np.linspace(0, 1, N), (-1, 1))
    k = OrthogonalRBFKernel(
        gpflow.kernels.RBF(lengthscales=10), EmpiricalMeasure(location)
    )

    mu = np.zeros(N)
    f = np.random.multivariate_normal(mu, k.K(location), size=1)
    np.testing.assert_almost_equal(f.mean(), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY)


def test_EmpiricalMeasure_with_weights():
    np.random.seed(44)
    N = 10
    location = np.reshape(np.linspace(0, 1, N), (-1, 1))
    weights = np.random.randn(N, 1)
    weights /= weights.sum()
    k = OrthogonalRBFKernel(
        gpflow.kernels.RBF(lengthscales=10), EmpiricalMeasure(location, weights)
    )
    mu = np.zeros(N)
    f = np.random.multivariate_normal(mu, k.K(location), size=1)
    np.testing.assert_almost_equal(
        np.dot(f, weights).mean(), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY
    )


def test_MOGMeasure():
    np.random.seed(44)
    K = 5
    N = 10
    means = np.random.randn(K)
    weights = np.random.rand(K)
    weights /= weights.sum()
    variances = np.random.rand(K) + 0.1
    k = OrthogonalRBFKernel(
        gpflow.kernels.RBF(lengthscales=10), MOGMeasure(means, variances, weights)
    )

    # sample from the MOG
    xx = np.random.randn(N, K) * np.sqrt(variances) + means
    index = np.random.multinomial(1, weights, N).argmax(1)
    xx = xx[np.arange(N), index]

    # sample from the GP
    mu = np.zeros(N)
    f = np.random.multivariate_normal(mu, k.K(xx.reshape(-1, 1)), size=1)
    np.testing.assert_almost_equal(f.mean(), 0.0, decimal=_THRESHOLD_NUMERICAL_ACCURACY)


def test_MOGMeasure_equivalence_to_GaussianMeasure():
    mu = np.array([3.0, 3.0])
    var = np.array([5.0, 5.0])
    lengthscales = 10.0
    weights = np.array([0.2, 0.8])
    k_gmm = OrthogonalRBFKernel(
        gpflow.kernels.RBF(lengthscales=lengthscales), MOGMeasure(mu, var, weights)
    )
    k_gaussian = OrthogonalRBFKernel(
        gpflow.kernels.RBF(lengthscales=lengthscales), GaussianMeasure(3, 5)
    )

    xx = np.array([[-2], [2.0], [3.0]])
    np.testing.assert_allclose(k_gaussian.K(xx), k_gmm.K(xx))


def test_gmm_fit():
    X = np.array([1.0, 1, 1, 10, 10, 10])
    measure = estimate_one_dim_gmm(K=2, X=X)
    np.testing.assert_almost_equal(np.sort(measure.means), np.array([1.0, 10.0]))

