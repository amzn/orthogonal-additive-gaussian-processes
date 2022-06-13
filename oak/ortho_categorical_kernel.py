# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.utilities import positive
from typing import List


# -

class OrthogonalCategorical(gpflow.kernels.Kernel):
    """
    :param p: list of probability measure for categorical kernels which sums up to 1, p_i = Prob(X = i)
    :param rank: the number of degrees of correlation between the outputs, see details in coregion kernel https://gpflow.readthedocs.io/en/master/_modules/gpflow/kernels/misc.html#Coregion
    :param active_dims: active dimension of input to apply this kernel to
    :return Constrained categorical kernel
    """

    def __init__(self, p: List, rank: int = 2, active_dims: int = None):
        super().__init__(active_dims=active_dims)
        num_cat = len(p)
        self.num_cat = num_cat
        self.p = p
        self.variance = gpflow.Parameter(1.0, transform=positive())
        W = tf.random.uniform(shape=[num_cat, rank])
        kappa = tf.ones(self.num_cat)
        # kappa = tf.zeros(self.num_cat)
        self.W = Parameter(W)
        self.kappa = Parameter(kappa, transform=positive())

    def output_covariance(self):
        A = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(
            self.kappa
        )
        Ap = tf.linalg.matmul(A, self.p)
        B = A - tf.linalg.matmul(Ap, Ap, transpose_b=True) / (
            tf.linalg.matmul(self.p, Ap, transpose_a=True)[0]
        )
        return B * self.variance

    def output_variance(self):
        A = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(
            self.kappa
        )
        Ap = tf.linalg.matmul(A, self.p)
        A_diag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        B_diag = A_diag - tf.reduce_sum(tf.square(Ap), 1) / (
            tf.linalg.matmul(self.p, Ap, transpose_a=True)[0]
        )
        return B_diag * self.variance

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
