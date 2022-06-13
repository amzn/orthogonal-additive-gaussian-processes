# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import pytest
import oak.normalising_flow as normalising_flow
from unittest import mock
from oak.normalising_flow import Normalizer


# -

# mock is used to test figures
@mock.patch("%s.normalising_flow.plt" % __name__)
def test_normalising_flow(mock_plt):
    np.random.seed(44)
    N = 100
    x = np.random.normal(2, 0.5, size=(N, 1))

    # apply normalising flow on x to check it is transformed to N(0,1)
    n = Normalizer(x, log=False)
    kl_before_optimising = n.KL_objective()

    opt = gpflow.optimizers.Scipy()
    opt.minimize(n.KL_objective, n.trainable_variables)

    y = n.bijector(x).numpy()
    # check transformed x has mean = 0 and var = 1
    np.testing.assert_almost_equal(0, y.mean(), decimal=2)
    np.testing.assert_almost_equal(1, y.std(), decimal=2)

    # check Kolmogorov-Smirnov test https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test do not
    # reject null hypothesis that transformed x and N(0,1) are identical at 5% significant level
    s, pvalue = n.kstest()
    assert pvalue > 0.05

    # test KL objective has decreased
    kl_after_optimising = n.KL_objective()
    assert kl_after_optimising < kl_before_optimising

    # Assert plt.figure got called
    n.plot(title="NF")
    assert mock_plt.figure.called

    # Assert plt.title has been called with expected arg
    mock_plt.title.assert_called_once_with("NF")
