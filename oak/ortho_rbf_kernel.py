# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import tensorflow as tf
from typing import Optional
from oak.input_measures import (
    Measure,
    EmpiricalMeasure,
    GaussianMeasure,
    MOGMeasure,
    UniformMeasure,
)


# -

class OrthogonalRBFKernel(gpflow.kernels.Kernel):
    """
    :param base_kernel: base RBF kernel before applying orthogonality constraint
    :param measure: input measure
    :param active_dims: active dimension
    :return: constrained BRF kernel
    """

    def __init__(
        self, base_kernel: gpflow.kernels.RBF, measure: Measure, active_dims=None
    ):
        super().__init__(active_dims=active_dims)
        self.base_kernel, self.measure = base_kernel, measure
        self.active_dims = self.active_dims
        if not isinstance(base_kernel, gpflow.kernels.RBF):
            raise NotImplementedError
        if not isinstance(
            measure,
            (
                UniformMeasure,
                GaussianMeasure,
                EmpiricalMeasure,
                MOGMeasure,
            ),
        ):
            raise NotImplementedError

        if isinstance(self.measure, UniformMeasure):

            def cov_X_s(X):
                tf.debugging.assert_shapes([(X, ("N", 1))])
                l = self.base_kernel.lengthscales
                sigma2 = self.base_kernel.variance
                return (
                    sigma2
                    * l
                    / (self.measure.b - self.measure.a)
                    * np.sqrt(np.pi / 2)
                    * (
                        tf.math.erf((self.measure.b - X) / np.sqrt(2) / l)
                        - tf.math.erf((self.measure.a - X) / np.sqrt(2) / l)
                    )
                )

            def var_s():
                l = self.base_kernel.lengthscales
                sigma2 = self.base_kernel.variance
                y = (self.measure.b - self.measure.a) / np.sqrt(2) / l
                return (
                    2.0
                    / ((self.measure.b - self.measure.a) ** 2)
                    * sigma2
                    * l ** 2
                    * (
                        np.sqrt(np.pi) * y * tf.math.erf(y)
                        + tf.exp(-tf.square(y))
                        - 1.0
                    )
                )

        if isinstance(self.measure, GaussianMeasure):

            def cov_X_s(X):
                tf.debugging.assert_shapes([(X, (..., "N", 1))])
                l = self.base_kernel.lengthscales
                sigma2 = self.base_kernel.variance
                mu, var = self.measure.mu, self.measure.var
                return (
                    sigma2
                    * l
                    / tf.sqrt(l ** 2 + var)
                    * tf.exp(-0.5 * ((X - mu) ** 2) / (l ** 2 + var))
                )

            def var_s():
                l = self.base_kernel.lengthscales
                sigma2 = self.base_kernel.variance
                return sigma2 * l / tf.sqrt(l ** 2 + 2 * self.measure.var)

        if isinstance(self.measure, EmpiricalMeasure):

            def cov_X_s(X):
                location = self.measure.location
                weights = self.measure.weights
                tf.debugging.assert_shapes(
                    [(X, ("N", 1)), (location, ("M", 1)), (weights, ("M", 1))]
                )
                return tf.matmul(self.base_kernel(X, location), weights)

            def var_s():
                location = self.measure.location
                weights = self.measure.weights
                tf.debugging.assert_shapes([(location, ("M", 1)), (weights, ("M", 1))])
                return tf.squeeze(
                    tf.matmul(
                        tf.matmul(
                            weights, self.base_kernel(location), transpose_a=True
                        ),
                        weights,
                    )
                )

        if isinstance(self.measure, MOGMeasure):

            def cov_X_s(X):
                tf.debugging.assert_shapes([(X, ("N", 1))])
                l = self.base_kernel.lengthscales
                sigma2 = self.base_kernel.variance
                mu, var, weights = (
                    self.measure.means,
                    self.measure.variances,
                    self.measure.weights,
                )
                tmp = tf.exp(-0.5 * ((X - mu) ** 2) / (l ** 2 + var)) / tf.sqrt(
                    l ** 2 + var
                )

                return sigma2 * l * tf.matmul(tmp, tf.reshape(weights, (-1, 1)))

            def var_s():
                l = self.base_kernel.lengthscales

                sigma2 = self.base_kernel.variance
                mu, var, w = (
                    self.measure.means,
                    self.measure.variances,
                    self.measure.weights,
                )
                dists = tf.square(mu[:, None] - mu[None, :])
                scales = tf.square(l) + var[:, None] + var[None, :]
                tmp = sigma2 * l / tf.sqrt(scales) * tf.exp(-0.5 * dists / scales)

                return tf.squeeze(tf.matmul(tf.matmul(w[None, :], tmp), w[:, None]))

        self.cov_X_s = cov_X_s
        self.var_s = var_s

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        :param X: input array X
        :param X2: input array X2, if None, set to X
        :return: kernel matrix K(X,X2)
        """
        cov_X_s = self.cov_X_s(X)
        if X2 is None:
            cov_X2_s = cov_X_s
        else:
            cov_X2_s = self.cov_X_s(X2)
        k = (
            self.base_kernel(X, X2)
            - tf.tensordot(cov_X_s, tf.transpose(cov_X2_s), 1) / self.var_s()
        )
        return k

    def K_diag(self, X):
        cov_X_s = self.cov_X_s(X)
        k = self.base_kernel.K_diag(X) - tf.square(cov_X_s[:, 0]) / self.var_s()
        return k
