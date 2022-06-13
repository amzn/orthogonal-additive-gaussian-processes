# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests kernel behave as expected
"""
from functools import reduce
from itertools import combinations
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
from oak.oak_kernel import OAKKernel
from oak.ortho_binary_kernel import OrthogonalBinary
from oak.ortho_rbf_kernel import OrthogonalRBFKernel


@pytest.mark.parametrize(
    "kernel",
    (
        OAKKernel(
            [gpflow.kernels.RBF],
            num_dims=1,
            max_interaction_depth=1,
        ),
        OAKKernel(
            [gpflow.kernels.RBF],
            num_dims=1,
            max_interaction_depth=1,
            constrain_orthogonal=True,
        ),
        OrthogonalBinary(),
        OrthogonalRBFKernel(gpflow.kernels.RBF(), GaussianMeasure(0, 1)),
        OrthogonalRBFKernel(gpflow.kernels.RBF(), UniformMeasure(0, 1)),
        OrthogonalRBFKernel(
            gpflow.kernels.RBF(), EmpiricalMeasure(np.array([[0.1], [0.5], [0.5]]))
        ),
        OrthogonalRBFKernel(
            gpflow.kernels.RBF(), GaussianMeasure(0, 1),
        ),
        OrthogonalRBFKernel(
            gpflow.kernels.RBF(),
            MOGMeasure(
                np.array([3.0, 2.0]), np.array([3.0, 10.0]), np.array([0.6, 0.4])
            ),
        ),
    ),
)
def test_kernel_1d(kernel: gpflow.kernels.Kernel):
    X = np.array([[0.1], [0.5], [0.5]])
    np.testing.assert_allclose(
        np.diag(kernel.K(X, X)),
        kernel.K_diag(X),
        err_msg="diagonal calculation is not correct",
    )
    np.testing.assert_allclose(
        kernel.K(X, X), kernel(X, X), err_msg="k and k.K not the same"
    )


@pytest.mark.parametrize("num_dims", [3, 4])
def test_newton_girard(num_dims):
    k = OAKKernel(
        [gpflow.kernels.RBF for i in range(num_dims)],
        num_dims=num_dims,
        max_interaction_depth=num_dims,
    )
    xx = [np.random.randn(2, 2) for _ in range(num_dims)]
    result = k.compute_additive_terms(xx)

    # compute the result the hard way
    result_hard = [np.ones((2, 2))] + [
        reduce(np.add, map(lambda x: np.prod(x, axis=0), combinations(xx, i)))
        for i in range(1, num_dims)
    ]

    for r1, r2 in zip(result, result_hard):
        np.testing.assert_allclose(r1, r2)


@pytest.mark.parametrize("active_dims", [[0], [1]])
@pytest.mark.parametrize(
    "measure",
    (
        GaussianMeasure(0, 1),
        UniformMeasure(0, 1),
        EmpiricalMeasure(np.array([[0.1], [0.5]])),
        MOGMeasure(np.array([3.0, 2.0]), np.array([3.0, 10.0]), np.array([0.6, 0.4])),
        MOGMeasure(
            np.array([3, 2], dtype=int),
            np.array([3, 10], dtype=int),
            np.array([0.6, 0.4]),
        ),
    ),
)
def test_orthogonal_rbf_kernel_2d_with_active_dims(active_dims, measure):
    k = OrthogonalRBFKernel(
        gpflow.kernels.RBF(lengthscales=10), measure, active_dims=active_dims
    )
    X = np.array([[0.1, 0.2], [0.5, 0.5], [0.5, 0.7]])
    np.testing.assert_allclose(
        np.diag(k.K(X[:, active_dims], X[:, active_dims])),
        k.K_diag(X[:, active_dims]),
        err_msg="diagonal calculation is not correct",
    )
    np.testing.assert_allclose(
        k.K(X[:, active_dims]), k(X, X), err_msg="k and k.K not the same"
    )
