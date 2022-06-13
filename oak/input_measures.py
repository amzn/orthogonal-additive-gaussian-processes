# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
# -

import numpy as np
import tensorflow as tf

"""
Input measure
"""


class Measure:
    pass


class UniformMeasure(Measure):
    """
    :param a: lower bound of the uniform distribution
    :param b: upper bound of the uniform distribution
    :return: Uniform measure for inputs
    """

    def __init__(self, a: float, b: float):
        self.a, self.b = a, b


class GaussianMeasure(Measure):
    """
    :param mu: Mean of Gaussian measure
    :param var: variance of Gaussian measure
    :return: Gaussian measure for inputs
    """

    def __init__(self, mu: float, var: float):
        self.mu, self.var = mu, var


class EmpiricalMeasure(Measure):
    """
    :param location: location of the input data
    :param weights: weights on the location of the data
    :return: Empirical dirac measure for inputs with weights on the locations
    """

    def __init__(self, location: np.ndarray, weights: Optional[np.ndarray] = None):
        self.location = location
        if weights is None:
            weights = 1 / len(location) * np.ones((location.shape[0], 1))
        assert np.isclose(
            weights.sum(), 1.0, atol=1e-6
        ), f"not close to 1 {weights.sum()}"
        self.weights = weights


class MOGMeasure(Measure):
    """
    :param means: mean of the Gaussian measures
    :param variances: variances of the Gaussian measures
    :param weights: weights on the Gaussian measures
    :return: mixture of Gaussian measure
    """

    def __init__(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray):
        tf.debugging.assert_shapes(
            [(means, ("K",)), (variances, ("K",)), (weights, ("K",))]
        )
        assert np.isclose(
            weights.sum(), 1.0, atol=1e-6
        ), f"Weights not close to 1 {weights.sum()}"
        self.means, self.variances, self.weights = (
            means.astype(float),
            variances.astype(float),
            weights,
        )
