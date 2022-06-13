# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
from oak.ortho_categorical_kernel import OrthogonalCategorical
# -

_THRESHOLD_NUMERICAL_ACCURACY = 1e-3


def test_OrthogonalCategorical():
    np.random.seed(44)
    tf.random.set_seed(44)
    N = 1000
    num_cat = 2
    p = np.ones((num_cat, 1)) / num_cat
    k = OrthogonalCategorical(p, rank=2, active_dims=[0])
    xx = np.reshape(np.random.choice(num_cat, N, p=p[:, 0]), (-1, 1))
    mu = np.zeros(N)
    f = np.random.multivariate_normal(mu, k.K(xx), size=2000)
    assert np.abs(f.mean()) < _THRESHOLD_NUMERICAL_ACCURACY
