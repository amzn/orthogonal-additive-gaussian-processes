# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive


# -

class OrthogonalBinary(gpflow.kernels.Kernel):
    """
    :param p0: probability of binary measure
    :param active_dims: active dimension along which the kernel is to be applied
    :return: constrained binary kernel
    """

    def __init__(
        self,
        p0: float = 0.5,
        active_dims: int = None,
    ):
        super().__init__(active_dims=active_dims)
        self.variance = gpflow.Parameter(1.0, transform=positive())
        self.p0 = p0

    def output_covariance(self):
        p0 = self.p0
        p1 = 1.0 - p0
        B = np.array([[np.square(p1), -p0 * p1], [-p0 * p1, np.square(p0)]])
        return B * self.variance

    def output_variance(self):
        p0 = self.p0
        p1 = 1.0 - p0
        return np.array([np.square(p1), np.square(p0)]) * self.variance

    def K(self, X, X2=None):
        shape_constraints = [
            (X, [..., "N", 1]),
        ]
        if X2 is not None:
            shape_constraints.append((X2, [..., "M", 1]))
        tf.debugging.assert_shapes(shape_constraints)
        X = tf.cast(X[..., 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[..., 0], tf.int32)
        B = self.output_covariance()
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    def K_diag(self, X):
        tf.debugging.assert_shapes([(X, [..., "N", 1])])
        X = tf.cast(X[..., 0], tf.int32)
        B_diag = self.output_variance()
        return tf.gather(B_diag, X)
