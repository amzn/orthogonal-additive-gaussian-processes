# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow import set_trainable
from gpflow.models import SGPR
from sklearn.cluster import KMeans
# -

from oak.input_measures import GaussianMeasure
from oak.ortho_binary_kernel import OrthogonalBinary
from oak.utils import get_model_sufficient_statistics

opt = gpflow.optimizers.Scipy()


from oak.utils import (
    compute_L_binary_kernel,
    compute_sobol,
    f1,
    f2,
    f4,
    grammer_to_kernel,
    model_to_kernel_list,
)

TOL = 1e-3


def test_f1():

    sigma = 1
    delta = 1
    lengthscales = 1
    mu = 0

    def kk1(x, y):
        return sigma ** 2 * np.exp(-((x - y) ** 2) / (2 * lengthscales ** 2))

    def kk2(x, y):
        return (
            sigma ** 2
            * lengthscales
            * np.sqrt(lengthscales ** 2 + 2 * delta ** 2)
            / (lengthscales ** 2 + delta ** 2)
            * np.exp(
                -((x - mu) ** 2 + (y - mu) ** 2)
                / (2 * (lengthscales ** 2 + delta ** 2))
            )
        )

    np.random.seed(44)
    tf.random.set_seed(44)
    Z = np.random.normal(0, 1, (2, 1))

    zz = np.random.normal(0, delta, size=(100000, 1))

    y_numerical = np.mean(
        [kk1(Z[0, :], zz[jj]) * kk1(Z[1, :], zz[jj]) for jj in range(100000)]
    )
    y = f1(Z[0, :], Z[1, :], sigma, lengthscales, delta, mu)

    print(np.abs(y_numerical - y))
    assert np.abs(y_numerical - y) < TOL


def test_f2():

    sigma = 1
    delta = 1
    lengthscales = 1
    mu = 0

    def kk1(x, y):
        return sigma ** 2 * np.exp(-((x - y) ** 2) / (2 * lengthscales ** 2))

    def kk2(x, y):
        return (
            sigma ** 2
            * lengthscales
            * np.sqrt(lengthscales ** 2 + 2 * delta ** 2)
            / (lengthscales ** 2 + delta ** 2)
            * np.exp(
                -((x - mu) ** 2 + (y - mu) ** 2)
                / (2 * (lengthscales ** 2 + delta ** 2))
            )
        )

    np.random.seed(44)
    tf.random.set_seed(44)
    Z = np.random.normal(0, 1, (2, 1))

    zz = np.random.normal(0, delta, size=(100000, 1))

    y_numerical = np.mean(
        [kk1(Z[0, :], zz[jj]) * kk2(Z[1, :], zz[jj]) for jj in range(100000)]
    )
    y = f2(Z[0, :], Z[1, :], sigma, lengthscales, delta, mu)

    assert np.abs(y_numerical - y) < TOL


def test_f4():

    sigma = 1
    delta = 1
    lengthscales = 1
    mu = 0

    def kk1(x, y):
        return sigma ** 2 * np.exp(-((x - y) ** 2) / (2 * lengthscales ** 2))

    def kk2(x, y):
        return (
            sigma ** 2
            * lengthscales
            * np.sqrt(lengthscales ** 2 + 2 * delta ** 2)
            / (lengthscales ** 2 + delta ** 2)
            * np.exp(
                -((x - mu) ** 2 + (y - mu) ** 2)
                / (2 * (lengthscales ** 2 + delta ** 2))
            )
        )

    np.random.seed(44)
    tf.random.set_seed(44)
    Z = np.random.normal(0, 1, (2, 1))

    zz = np.random.normal(0, delta, size=(100000, 1))

    y_numerical = np.mean(
        [kk2(Z[0, :], zz[jj]) * kk2(Z[1, :], zz[jj]) for jj in range(100000)]
    )
    y = f4(Z[0, :], Z[1, :], sigma, lengthscales, delta, mu)

    assert np.abs(y_numerical - y) < TOL



@pytest.mark.skip(
    reason="too slow a test takes about 30 seconds covered by test_sobol_indices which is faster"
)
def test_compute_sobol():

    selected_dims = [[0], [1], [0, 1]]
    P_dims = []
    offset = 0
    delta = 1
    mu = 0

    X_train = np.random.normal(0, delta, (500, 2))
    Y_train = np.reshape(
        X_train[:, 0] ** 2 + X_train[:, 1] * 2 + X_train[:, 0] * X_train[:, 1], (-1, 1)
    )

    kmeans = KMeans(n_clusters=500, random_state=0).fit(X_train)
    Z = kmeans.cluster_centers_

    data = (X_train, Y_train)

    sgpr = SGPR(
        data,
        kernel=np.sum(
            grammer_to_kernel(
                selected_dims, P_dims, offset, GaussianMeasure(mu, delta ** 2)
            )
        ),
        inducing_variable=Z,
    )

    set_trainable(sgpr.inducing_variable, False)

    opt.minimize(sgpr.training_loss, sgpr.trainable_variables, method="BFGS")

    alpha = get_model_sufficient_statistics(sgpr, get_L=False)
    kernel_list = model_to_kernel_list(sgpr, selected_dims)
    sobol = compute_sobol(sgpr, kernel_list, delta, mu, alpha)

    assert np.abs(sobol - np.array([2, 4, 1])).max() < TOL


# try a few values of p including the corner cases
@pytest.mark.parametrize("p", (0.0, 0.77, 1.0))
def test_compute_L_binary_kernel(p: float):

    TOL = 1e-16

    # this test verify the calculation of L in Appendix G.1 is correct.

    # generate training data X
    X = tf.convert_to_tensor(np.reshape(np.random.binomial(1, p, 1000), (-1, 1)))
    # use compute_L_binary_kernel function to calculate the L matrix
    L = compute_L_binary_kernel(X, p, 1, 0)

    # calculate L using the binary kernel directly
    x0 = np.reshape(0, (-1, 1))
    x1 = np.reshape(1, (-1, 1))

    K = OrthogonalBinary(p0=p, active_dims=[0])

    L1 = np.matmul(K(X, x0), K(x0, X)) * p + np.matmul(K(X, x1), K(x1, X)) * (1 - p)

    print(np.max(L - L1))
    assert np.max(L - L1) < TOL

