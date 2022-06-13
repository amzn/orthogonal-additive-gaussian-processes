# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow import set_trainable
from gpflow.models import GPR, SGPR
from sklearn.cluster import KMeans
from oak.input_measures import GaussianMeasure
from oak.model_utils import create_model_oak, oak_model
from oak.oak_kernel import get_list_representation
from oak.ortho_rbf_kernel import EmpiricalMeasure, OrthogonalRBFKernel
from oak.utils import (
    compute_L_empirical_measure,
    compute_sobol,
    compute_sobol_oak,
    get_model_sufficient_statistics,
    get_prediction_component,
    grammer_to_kernel,
    initialize_kmeans_with_binary,
    model_to_kernel_list,
)


# -

@pytest.mark.parametrize("is_oak_kernel", (False, True))
@pytest.mark.parametrize("is_sgpr", (False, True))
@pytest.mark.parametrize("lengthscale_bounds", [[1e-6, 100], None])
@pytest.mark.parametrize("share_var_across_orders", [True, False])
def test_compute_sobol(
    is_sgpr: bool,
    is_oak_kernel: bool,
    lengthscale_bounds: list,
    share_var_across_orders: bool,
):
    selected_dims = [[0], [1], [0, 1]]
    offset = 0
    delta = 1
    mu = 0
    N = 500
    X_train = np.random.normal(0, delta, (N, 2))
    Y_train = np.reshape(
        X_train[:, 0] ** 2 + X_train[:, 1] * 2 + X_train[:, 0] * X_train[:, 1], (-1, 1)
    )
    if is_sgpr:
        kmeans = KMeans(n_clusters=300, random_state=0).fit(X_train)
        Z = kmeans.cluster_centers_
    else:
        Z = None  # GPR model
    data = (X_train, Y_train)

    if is_oak_kernel:
        model = create_model_oak(
            data,
            inducing_pts=Z,
            optimise=False,
            zfixed=False,
            lengthscale_bounds=lengthscale_bounds,
            share_var_across_orders=share_var_across_orders,
        )
        if share_var_across_orders:
            model.kernel.variances[0].assign(0.76)
            model.kernel.variances[1].assign(96.935)
            model.kernel.variances[2].assign(128.27)
        else:
            model.kernel.variances[0].assign(0.01)
            model.kernel.kernels[0].base_kernel.variance.assign(1)
            model.kernel.kernels[1].base_kernel.variance.assign(1)
        model.kernel.kernels[0].base_kernel.lengthscales.assign(2.91)
        model.kernel.kernels[1].base_kernel.lengthscales.assign(9.20)

        model_indices, sobol = compute_sobol_oak(
            model,
            delta,
            mu,
            share_var_across_orders=share_var_across_orders,
        )
        assert len(model_indices) == len(sobol)
        assert model_indices == selected_dims
    else:
        if is_sgpr:
            model = SGPR(
                data,
                kernel=np.sum(
                    grammer_to_kernel(
                        selected_dims, offset, GaussianMeasure(mu, delta ** 2)
                    )
                ),
                inducing_variable=Z,
            )
            set_trainable(model.inducing_variable, False)
            sparse_gp = True
        else:
            model = GPR(
                data,
                kernel=np.sum(
                    grammer_to_kernel(
                        selected_dims, offset, GaussianMeasure(mu, delta ** 2)
                    )
                ),
            )
            sparse_gp = False
        model.kernel.kernels[0].base_kernel.variance.assign(99.98)
        model.kernel.kernels[0].base_kernel.lengthscales.assign(2.75)
        model.kernel.kernels[1].base_kernel.variance.assign(99.99)
        model.kernel.kernels[1].base_kernel.lengthscales.assign(99.99)
        # interaction term
        model.kernel.kernels[2].kernels[0].base_kernel.lengthscales.assign(4.762)
        model.kernel.kernels[2].kernels[0].base_kernel.variance.assign(99.99)
        model.kernel.kernels[2].kernels[1].base_kernel.lengthscales.assign(4.499)
        model.kernel.kernels[2].kernels[1].base_kernel.variance.assign(1.00)
        # constant term
        model.kernel.kernels[3].variance.assign(1.00)
        model.likelihood.variance.assign(1e-5)
        kernel_list = model_to_kernel_list(model, selected_dims)
        alpha = get_model_sufficient_statistics(model, get_L=False)
        sobol = compute_sobol(model, kernel_list, delta, mu, alpha, sparse_gp=sparse_gp)
    gpflow.utilities.print_summary(model)
    np.testing.assert_array_almost_equal(
        sobol, np.array([2, 4, 1], dtype=float), decimal=1
    )


def test_sobol_empirical_measure():
    x = np.random.normal(0, 1, (10, 1))
    y = x ** 2 + np.cos(x) + np.random.normal(0, 0.1, (10, 1))
    kernel = OrthogonalRBFKernel(
        gpflow.kernels.RBF(),
        EmpiricalMeasure(x, np.ones(x.shape) / 10),
        active_dims=[0],
    )

    m = GPR((x, y), kernel=kernel)

    var_samples = np.var(m.predict_f(x)[0].numpy())
    alpha = get_model_sufficient_statistics(m, get_L=False)
    L = compute_L_empirical_measure(
        tf.reshape(m.kernel.measure.location, [-1, 1]),
        m.kernel.measure.weights,
        m.kernel,
        x,
    )
    var_sobol = tf.tensordot(
        tf.tensordot(tf.transpose(alpha), L, axes=1), alpha, axes=1
    ).numpy()[0][0]
    gpflow.utilities.print_summary(m)
    np.testing.assert_array_almost_equal(var_samples, var_sobol, decimal=5)


@pytest.mark.parametrize("empirical_measure", [[0], [0, 1]])
@pytest.mark.parametrize("share_var_across_orders", [True, False])
def test_sobol_oak_kernel_empirical(
    empirical_measure: List[float], share_var_across_orders: bool
):
    # test Sobol with empirical measure works with the model API, the test compares
    # calculated Sobol with empirical variance of function components
    np.random.seed(44)
    n = 100
    m = 50
    X = np.random.normal(0, 1, (n, 2))
    y = np.reshape(X[:, 0] ** 2 + X[:, 1] * 2 + X[:, 0] * X[:, 1], (-1, 1))
    oak = oak_model(
        max_interaction_depth=X.shape[1],
        num_inducing=m,
        sparse=True,
        empirical_measure=empirical_measure,
        share_var_across_orders=share_var_across_orders,
    )
    oak.fit(X, y, optimise=False)
    # save hyperparameters to reduce computation time
    oak.m.kernel.kernels[0].base_kernel.lengthscales.assign(2)
    oak.m.kernel.kernels[1].base_kernel.lengthscales.assign(5)
    if share_var_across_orders:
        oak.m.kernel.variances[0].assign(1e-3)
        oak.m.kernel.variances[1].assign(90)
        oak.m.kernel.variances[2].assign(15)
    gpflow.utilities.print_summary(oak.m)

    oak.get_sobol()
    alpha = get_model_sufficient_statistics(oak.m, get_L=False)
    prediction_list = get_prediction_component(
        oak.m,
        alpha,
        oak._transform_x(X),
        share_var_across_orders=share_var_across_orders,
    )
    # calculate variance of each functional component with empirical data
    var_samples = np.array(
        [
            np.var(prediction_list[0].numpy()),
            np.var(prediction_list[1].numpy()),
            np.var(prediction_list[2].numpy()),
        ]
    )
    var_samples = var_samples / var_samples.sum()
    np.testing.assert_array_almost_equal(var_samples, oak.normalised_sobols, decimal=1)

@pytest.mark.parametrize("num_dims", (2, 7))
@pytest.mark.parametrize("zfixed", (True, False))
@pytest.mark.parametrize("normalisation", (True, False))
@pytest.mark.parametrize("share_var_across_orders", (True, False))
def test_sobol_indices(
    num_dims,
    zfixed,
    concrete_normalised_10_rows_data,
    normalisation,
    share_var_across_orders,
):
    X, y = concrete_normalised_10_rows_data
    max_interaction_depth = 2
    optimise = False
    Z = X[:3, :]

    def create_sgpr(X, y):
        sgpr = create_model_oak(
            (X, y),
            max_interaction_depth=max_interaction_depth,
            constrain_orthogonal=True,
            inducing_pts=Z,
            optimise=optimise,
            zfixed=zfixed,
        )
        selected_dims, kernel_list = get_list_representation(
            sgpr.kernel, num_dims=num_dims
        )
        alpha = get_model_sufficient_statistics(sgpr, get_L=False)
        return sgpr, selected_dims, kernel_list, alpha

    sgpr, selected_dims, kernel_list, alpha = create_sgpr(X, y)
    delta = 1
    mu = 0
    model_indices, sobol = compute_sobol_oak(
        sgpr,
        delta,
        mu,
        share_var_across_orders=share_var_across_orders,
    )
    assert np.all(np.array(sobol) > 0)


@pytest.mark.parametrize("is_sgpr", (False, True))
@pytest.mark.parametrize("both_binary", (False, True))
def test_compute_sobol_with_binary(is_sgpr: bool, both_binary: bool):
    # test sobol computation for two cases: 1) two binary kernel, 2) one continuous and one binary kernel
    delta = 1
    mu = 0
    N = 200

    p1 = 0.5
    np.random.seed(42)
    X1 = np.reshape(np.random.binomial(1, p1, N), (N, 1))

    selected_dims = [[0], [1], [0, 1]]

    if both_binary:
        p2 = 0.9
        X2 = np.reshape(np.random.binomial(1, p2, N), (N, 1))
    else:
        X2 = np.reshape(np.random.normal(mu, np.sqrt(delta), N), (N, 1))

    X_train = np.concatenate((X1, X2), 1).astype("float64")
    Y_train = np.reshape(
        X_train[:, 0]
        + X_train[:, 1]
        + X_train[:, 0] * X_train[:, 1]
        + np.random.normal(0, 0.1, N),
        (-1, 1),
    )

    Y_train = Y_train - Y_train.mean()
    if is_sgpr:
        Z = (
            initialize_kmeans_with_binary(X_train, binary_index=[0, 1], n_clusters=100)
            if both_binary
            else initialize_kmeans_with_binary(
                X_train, binary_index=[0], continuous_index=[1], n_clusters=100
            )
        )
    else:
        Z = None

    data = (X_train, Y_train)

    p0 = [1 - p1, 1 - p2] if both_binary else [1 - p1, None]

    model = create_model_oak(data, inducing_pts=Z, optimise=False, zfixed=True, p0=p0)

    if not both_binary:
        model.kernel.kernels[1].base_kernel.lengthscales.assign(9.20)

    model_indices, sobol = compute_sobol_oak(
        model,
        delta,
        mu,
    )

    assert len(model_indices) == len(sobol)
    assert model_indices == selected_dims
    assert np.all(np.array(sobol) >= 0)

    print(sobol)
    if both_binary:
        s1 = (1 + p2) ** 2 * p1 * (1 - p1)
        s2 = (1 + p1) ** 2 * p2 * (1 - p2)
        print(
            np.array(
                [
                    s1,
                    s2,
                    p1
                    - p1 ** 2
                    + p2
                    - p2 ** 2
                    + 5 * p1 * p2
                    - p1 ** 2 * p2 ** 2
                    - 2 * p1 ** 2 * p2
                    - 2 * p1 * p2 ** 2
                    - s1
                    - s2,
                ],
                dtype=float,
            )
        )
        np.testing.assert_array_almost_equal(
            sobol,
            np.array(
                [
                    s1,
                    s2,
                    p1
                    - p1 ** 2
                    + p2
                    - p2 ** 2
                    + 5 * p1 * p2
                    - p1 ** 2 * p2 ** 2
                    - 2 * p1 ** 2 * p2
                    - 2 * p1 * p2 ** 2
                    - s1
                    - s2,
                ],
                dtype=float,
            ),
            decimal=1,
        )

    else:
        s1 = p1 * (1 - p1)
        s2 = delta * (1 + p1) ** 2
        print(
            np.array(
                [s1, s2, delta + p1 * (1 - p1) + 3 * p1 * delta - s1 - s2], dtype=float
            )
        )
        np.testing.assert_array_almost_equal(
            sobol,
            np.array(
                [s1, s2, delta + p1 * (1 - p1) + 3 * p1 * delta - s1 - s2], dtype=float
            ),
            decimal=1,
        )
