# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import pytest
from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPR, SGPR
from oak.input_measures import GaussianMeasure
from oak.model_utils import create_model_oak
from oak.ortho_rbf_kernel import OrthogonalRBFKernel


# -

@pytest.mark.parametrize("num_inducings", [0, 2])
def test_OrthogonalRBFKernel_optimisation(num_inducings):
    data = [[0.0], [1.0], [2.0]]
    X = np.array(data)
    y = np.array(data)[:, 0].reshape(-1, 1)  # 1-D output
    Z = X[:num_inducings, :] if num_inducings > 0 else None

    k = OrthogonalRBFKernel(
        gpflow.kernels.RBF(lengthscales=10),
        GaussianMeasure(0, 1),
    )

    if Z is not None:
        model = SGPR((X, y), kernel=k, inducing_variable=InducingPoints(Z))
    else:
        model = GPR((X, y), kernel=k)

    initial_log_likelihood = model.maximum_log_likelihood_objective()
    assert not np.isnan(initial_log_likelihood)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss_closure(),
        model.trainable_variables,
        method="BFGS",
        compile=True,
        options=dict(disp=True, maxiter=2),
    )
    assert initial_log_likelihood < model.maximum_log_likelihood_objective()


@pytest.mark.parametrize("num_inducings", [0, 2])
def test_oak_optimisation(num_inducings):
    data = [[0.0], [1.0], [2.0]]
    X = np.array(data)
    y = np.array(data)[:, 0].reshape(-1, 1)  # 1-D output
    Z = X[:num_inducings, :] if num_inducings > 0 else None

    model = create_model_oak(
        (X, y), inducing_pts=Z, optimise=False, zfixed=True,
    )

    initial_log_likelihood = model.maximum_log_likelihood_objective()
    assert not np.isnan(initial_log_likelihood)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss_closure(),
        model.trainable_variables,
        method="BFGS",
        compile=True,
        options=dict(disp=True, maxiter=2),
    )
    assert initial_log_likelihood < model.maximum_log_likelihood_objective()
