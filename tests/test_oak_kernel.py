# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import pytest
# -

from oak.model_utils import create_model_oak
from oak.oak_kernel import KernelComponenent, OAKKernel, get_list_representation


@pytest.mark.parametrize(
    "data", [[[0.0], [1.0], [2.0]], [[0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]]
)
@pytest.mark.parametrize("num_inducings", [0, 2])
@pytest.mark.parametrize("lengthscale_bounds", [[1e-6, 2], None])
def test_oak(data, num_inducings, lengthscale_bounds):
    X = np.array(data)
    y = np.array(data)[:, 0].reshape(-1, 1)  # 1-D output
    Z = X[:num_inducings, :] if num_inducings > 0 else None
    model = create_model_oak(
        (X, y),
        inducing_pts=Z,
        lengthscale_bounds=lengthscale_bounds,
    )

    assert not np.isnan(model.maximum_log_likelihood_objective())


def test_kernel_component_constant(concrete_normalised_10_rows_data):
    X, y = concrete_normalised_10_rows_data
    x_try = X[:, 1][:, None]
    k = OAKKernel(
        [gpflow.kernels.RBF for i in range(x_try.shape[1])],
        num_dims=x_try.shape[1],
        max_interaction_depth=0,
        constrain_orthogonal=True,
    )
    k.variances[0].assign(0.3)
    np.testing.assert_allclose(
        k(x_try), KernelComponenent(k, [])(x_try), err_msg="0 order"
    )


def test_kernel_component_one_dimensional(
    concrete_normalised_10_rows_data
):
    X, y = concrete_normalised_10_rows_data
    x_try = X[:, 1][:, None]
    k = OAKKernel(
        [gpflow.kernels.RBF for i in range(x_try.shape[1])],
        num_dims=x_try.shape[1],
        max_interaction_depth=1,
        constrain_orthogonal=True,
    )
    k.variances[0].assign(0.3)
    k.variances[1].assign(3.3)
    np.testing.assert_allclose(
        k(x_try),
        KernelComponenent(k, [])(x_try) + KernelComponenent(k, [0])(x_try),
        err_msg="1 order 1-D",
    )


def test_kernel_component_two_dimensional_order_one_effects(
    concrete_normalised_10_rows_data
):
    X, y = concrete_normalised_10_rows_data
    x_try = X[:, :2]
    k = OAKKernel(
        [gpflow.kernels.RBF for i in range(x_try.shape[1])],
        num_dims=x_try.shape[1],
        max_interaction_depth=1,
        constrain_orthogonal=True,
    )
    np.testing.assert_allclose(
        k(x_try),
        KernelComponenent(k, [])(x_try)
        + KernelComponenent(k, [0])(x_try)
        + KernelComponenent(k, [1])(x_try),
        err_msg="1 order 2-D",
    )


def test_kernel_component_two_dimensional_order_two_effects(
    concrete_normalised_10_rows_data
):
    X, y = concrete_normalised_10_rows_data
    x_try = X[:, :2]
    k = OAKKernel(
        [gpflow.kernels.RBF for i in range(x_try.shape[1])],
        num_dims=x_try.shape[1],
        max_interaction_depth=2,
        constrain_orthogonal=True,
    )
    k.variances[0].assign(1.3)  # so we see if variance change
    k.variances[1].assign(3.3)
    k.variances[2].assign(4.3)
    np.testing.assert_allclose(
        k(x_try),
        KernelComponenent(k, [])(x_try)
        + KernelComponenent(k, [0])(x_try)
        + KernelComponenent(k, [1])(x_try)
        + KernelComponenent(k, [0, 1])(x_try),
        err_msg="2 order 2-D",
    )
    np.testing.assert_allclose(
        k.K_diag(x_try),
        KernelComponenent(k, []).K_diag(x_try)
        + KernelComponenent(k, [0]).K_diag(x_try)
        + KernelComponenent(k, [1]).K_diag(x_try)
        + KernelComponenent(k, [0, 1]).K_diag(x_try),
        err_msg="2 order 2-D K_diag",
    )


@pytest.mark.parametrize("num_dims", (2, 5, 7))
def test_get_list_representation_two_dimensional(
    num_dims, concrete_normalised_10_rows_data
):
    X, y = concrete_normalised_10_rows_data
    x_try = X[:, :2]
    k = OAKKernel(
        [gpflow.kernels.RBF for i in range(x_try.shape[1])],
        num_dims=x_try.shape[1],
        max_interaction_depth=2,
        constrain_orthogonal=True,
    )
    selected_dims, kernel_list = get_list_representation(k, num_dims=2)
    if num_dims == 2:
        assert selected_dims == [[], [0], [1], [0, 1]]
    assert len(kernel_list) == len(selected_dims)

    # checking K_Diag
    np.testing.assert_allclose(k.K_diag(X), np.diag(k(X)))
    k_el_diag = [l.K_diag(X) for l in kernel_list]
    np.testing.assert_allclose(k.K_diag(X), np.sum(k_el_diag, axis=0))

    # checking K
    K = k(X)
    k_el = [l(X) for l in kernel_list]
    np.testing.assert_allclose(K, np.sum(k_el, axis=0))
