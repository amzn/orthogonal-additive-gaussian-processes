# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import set_trainable
from gpflow.config import default_float, default_jitter
from gpflow.covariances.dispatch import Kuf, Kuu
from sklearn.cluster import KMeans
from gpflow.models import GPModel
from oak.input_measures import EmpiricalMeasure, GaussianMeasure, MOGMeasure
from oak.oak_kernel import (
    KernelComponenent,
    OAKKernel,
    bounded_param,
    get_list_representation,
)
from oak.ortho_binary_kernel import OrthogonalBinary
from oak.ortho_categorical_kernel import OrthogonalCategorical
from oak.ortho_rbf_kernel import OrthogonalRBFKernel
# -

opt = gpflow.optimizers.Scipy()
tfd = tfp.distributions
f64 = gpflow.utilities.to_default_float


def model_to_kernel_list(model: GPModel, selected_dims: List):
    # exact list of kernels from the OAK model
    kernel = []
    model_dims = extract_active_dims(model)
    for i in range(len(selected_dims)):
        for j in range(len(model.kernel.kernels) - 1):
            if model_dims[j] == selected_dims[i]:
                kernel.append(model.kernel.kernels[j])
    # append offset kernel
    kernel.append(model.kernel.kernels[-1])
    return kernel


def extract_active_dims(m):
    # exact list of active dimensions from the OAK model m
    active_dims = []
    for i in range(len(m.kernel.kernels) - 1):
        # interaction with product kernel
        if type(m.kernel.kernels[i]) == gpflow.kernels.base.Product:
            sub_m = m.kernel.kernels[i].kernels
            dims = []
            for j in range(len(sub_m)):
                dim = sub_m[j].active_dims
                dims.append(dim[0])
        else:
            dims = m.kernel.kernels[i].active_dims

        active_dims.append(list(dims))
    return active_dims


def grammer_to_kernel(
    selected_dims,
    offset,
    measure=GaussianMeasure(0, 10),
    lengthscales_lo=1e-3,
    lengthscales_hi=100,
    variance_lo=0.01,
    variance_hi=100,
):
    # construct list of kernels
    # selected_dims: list of kernel indices
    selected_kernels = []
    for i in range(len(selected_dims)):
        # loop through depth
        k_list = []
        for j in range(len(selected_dims[i])):

            lengthscales = np.random.uniform(low=lengthscales_lo, high=lengthscales_hi)
            variance = np.random.uniform(low=variance_lo, high=variance_hi)

            dim = selected_dims[i][j] + offset
            if isinstance(measure, EmpiricalMeasure):
                location = measure.location
                k = OrthogonalRBFKernel(
                    gpflow.kernels.RBF(lengthscales=lengthscales, variance=variance),
                    EmpiricalMeasure(np.reshape(location[:, dim], (-1, 1))),
                    active_dims=[dim],
                )
            else:
                k = OrthogonalRBFKernel(
                    gpflow.kernels.RBF(lengthscales=lengthscales, variance=variance),
                    measure,
                    active_dims=[dim],
                )
            k.base_kernel.lengthscales = bounded_param(
                lengthscales_lo, lengthscales_hi, lengthscales
            )
            k.base_kernel.variance = bounded_param(variance_lo, variance_hi, variance)
            if j > 0:
                k.base_kernel.variance.assign(1)
                set_trainable(k.base_kernel.variance, False)

            k_list.append(k)
        k = np.prod(k_list)
        selected_kernels.append(k)

    # add a constant kernel
    k0 = gpflow.kernels.Constant(variance=10)
    selected_kernels.append(k0)

    return selected_kernels


def f1(x, y, sigma, lengthscales, delta, mu):
    # eq (44) in Appendix G.1 of paper for calculating Sobol indices
    return (
        sigma ** 4
        * lengthscales
        / np.sqrt(lengthscales ** 2 + 2 * delta ** 2)
        * np.exp(-((x - y) ** 2) / (4 * lengthscales ** 2))
        * np.exp(-((mu - (x + y) / 2) ** 2) / (2 * delta ** 2 + lengthscales ** 2))
    )


def f2(x, y, sigma, lengthscales, delta, mu):
    # eq (45) in Appendix G.1 of paper for calculating Sobol indices
    M = 1 / (lengthscales ** 2) + 1 / (lengthscales ** 2 + delta ** 2)
    m = 1 / M * (mu / (lengthscales ** 2 + delta ** 2) + x / lengthscales ** 2)
    C = (
        x ** 2 / (lengthscales ** 2)
        + mu ** 2 / (lengthscales ** 2 + delta ** 2)
        - m ** 2 * M
    )
    return (
        sigma ** 4
        * lengthscales
        * np.sqrt((lengthscales ** 2 + 2 * delta ** 2) / (delta ** 2 * M + 1))
        * np.exp(-C / 2)
        / (lengthscales ** 2 + delta ** 2)
        * np.exp(-((y - mu) ** 2) / (2 * (lengthscales ** 2 + delta ** 2)))
        * np.exp(-((m - mu) ** 2) / (2 * (1 / M + delta ** 2)))
    )


def f3(x, y, sigma, lengthscales, delta, mu):
    # eq (46) in Appendix G.1 of paper for calculating Sobol indices
    return f2(y, x, sigma, lengthscales, delta, mu)


def f4(x, y, sigma, lengthscales, delta, mu):
    # eq (47) in Appendix G.1 of paper for calculating Sobol indices
    return (
        sigma ** 4
        * lengthscales ** 2
        * (lengthscales ** 2 + 2 * delta ** 2)
        * np.sqrt(
            (lengthscales ** 2 + delta ** 2) / (lengthscales ** 2 + 3 * delta ** 2)
        )
        / ((lengthscales ** 2 + delta ** 2) ** 2)
        * np.exp(
            -((x - mu) ** 2 + (y - mu) ** 2) / (2 * (lengthscales ** 2 + delta ** 2))
        )
    )


def get_model_sufficient_statistics(m, get_L=True):
    """
    Compute a vector "alpha" and a matrix "L" which can be used for easy prediction.
    """

    X_data, Y_data = m.data
    if isinstance(m, gpflow.models.SVGP):
        posterior = m.posterior()
        # details of Qinv can be found https://github.com/GPflow/GPflow/blob/develop/gpflow/posteriors.py
        alpha = posterior.alpha
        if get_L:
            L = tf.linalg.cholesky(tf.linalg.inv(posterior.Qinv[0]))
    elif isinstance(m, gpflow.models.SGPR):

        num_inducing = len(m.inducing_variable)
        err = Y_data - m.mean_function(X_data)
        kuf = Kuf(m.inducing_variable, m.kernel, X_data)
        kuu = Kuu(m.inducing_variable, m.kernel, jitter=default_jitter())

        sigma = tf.sqrt(m.likelihood.variance)
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma

        tmp1 = tf.linalg.solve(tf.transpose(LB), c)
        alpha = tf.linalg.solve(tf.transpose(L), tmp1)

        if get_L:
            # compute the effective L
            LAi = tf.linalg.triangular_solve(L, np.eye(L.shape[0]))
            LBiLAi = tf.linalg.triangular_solve(LB, LAi)
            L = tf.linalg.inv(LAi - LBiLAi)

    elif isinstance(m, gpflow.models.GPR):
        # prepare for prediction
        K = m.kernel(X_data)
        Ktilde = K + np.eye(X_data.shape[0]) * m.likelihood.variance
        L = np.linalg.cholesky(Ktilde)
        alpha = tf.linalg.cholesky_solve(L, Y_data)

    else:
        raise NotImplementedError
    if get_L:
        return alpha, L
    else:
        return alpha


def compute_L(
    X: tf.Tensor, lengthscale: float, variance: float, dim: int, delta: float, mu: float
) -> np.ndarray:
    # calculate the integral in eq (40) of Appendix G.1 in paper
    N = X.shape[0]
    sigma = np.sqrt(variance)
    xx = X[:, dim]
    yy = X[:, dim]

    x = np.repeat(xx, N)
    y = np.tile(yy, N)
    L = (
        f1(x, y, sigma, lengthscale, delta, mu)
        - f2(x, y, sigma, lengthscale, delta, mu)
        - f3(x, y, sigma, lengthscale, delta, mu)
        + f4(x, y, sigma, lengthscale, delta, mu)
    )
    L = np.reshape(L, (N, N))

    return L


def compute_L_binary_kernel(
    X: tf.Tensor, p0: float, variance: float, dim: int
) -> np.ndarray:

    """
    Compute L matrix needed for sobol index calculation for orthogonal binary kernels.
    :param X: training input tensor
    :param p0: probability measure for the data distribution (Prob(x=0))
    :param variance: variance parameter for the binary kernel, default is 1
    :param dim: active dimension of the kernel
    :return: sobol value L matrix

    """
    assert 0 <= p0 <= 1

    N = X.shape[0]
    xx = X[:, dim]
    yy = X[:, dim]

    x = np.repeat(xx, N)
    y = np.tile(yy, N)
    p1 = 1 - p0

    L = variance * (
        p0 * (p1 ** 2 * (1 - x) - p0 * p1 * x) * (p1 ** 2 * (1 - y) - p0 * p1 * y)
        + p1 * (-p0 * p1 * (1 - x) + p0 ** 2 * x) * (-p0 * p1 * (1 - y) + p0 ** 2 * y)
    )
    L = np.reshape(L, (N, N))

    return L


def compute_L_categorical_kernel(
    X: tf.Tensor, W: tf.Tensor, kappa: tf.Tensor, p: float, variance: float, dim: int
) -> np.ndarray:

    """
    Compute L matrix needed for sobol index calculation for orthogonal categorical kernels.
    :param X: training input tensor
    :param W: parameter of categorical kernel
    :param kappa: parameter of categorical kernel
    :param p: probability measure for the data distribution (Prob(x=0))
    :param variance: variance parameter for the categorical kernel, default is 1
    :param dim: active dimension of the kernel
    :return: sobol value L matrix

    """
    assert np.abs(p.sum() - 1) < 1e-6

    N = X.shape[0]

    A = tf.linalg.matmul(W, W, transpose_b=True) + tf.linalg.diag(kappa)
    Ap = tf.linalg.matmul(A, p)
    B = A - tf.linalg.matmul(Ap, Ap, transpose_b=True) / (
        tf.linalg.matmul(p, Ap, transpose_a=True)[0]
    )
    B = B * variance

    xx = tf.range(len(p), dtype=gpflow.config.default_float())

    K = tf.gather(
        tf.transpose(tf.gather(B, tf.cast(X[:, dim], tf.int32))), tf.cast(xx, tf.int32)
    )

    L = tf.linalg.matmul(K, K * p, transpose_a=True)

    return L


@tf.function
def compute_L_empirical_measure(
    x: tf.Tensor, w: tf.Tensor, kernel: OrthogonalRBFKernel, z: tf.Tensor
) -> np.ndarray:
    """
    Compute L matrix needed for sobol index calculation with empirical measure
    :param x: location of empirical measure
    :param w: weights of empirical measure, input density of the form 1/(\sum_i w_i) * \sum_i w_i (x==x_i)
    :param kernel: constrained kernel
    :param z: training data in full GP or inducing points locations in sparse GP
    :return: sobol value L matrix
    """

    # number of training/inducing points
    m = z.shape[0]
    # number of empirical locations
    n = x.shape[0]

    kxu = kernel.K(x, z)
    tf.debugging.assert_shapes([(kxu, (n, m))])
    w = tf.reshape(w, [1, n])
    L = tf.matmul(w * tf.transpose(kxu), kxu)

    return L


def compute_sobol_oak(
    model: gpflow.models.BayesianModel,
    delta: float,
    mu: float,
    share_var_across_orders: Optional[bool] = True,
) -> Tuple[List[List[int]], List[float]]:
    """
    Compute sobol indices for Duvenaud model
    :param model: gpflowm odel
    :param delta: prior variance of measure p(X)
    :param mu: prior mean of measure p(x)
    :param share_var_across_orders: whether to share the same variance across orders,
           if False, it uses original OrthogonalRBFKernel kernel \prod_i(1+k_i).
    :return: list of input dimension indices and list of sobol indices
    """
    print(model.kernel)
    assert isinstance(model.kernel, OAKKernel), "only work for OAK kernel"
    num_dims = model.data[0].shape[1]

    selected_dims_oak, kernel_list = get_list_representation(
        model.kernel, num_dims=num_dims
    )
    selected_dims_oak = selected_dims_oak[1:]  # skip constant term
    if isinstance(model, (gpflow.models.SGPR, gpflow.models.SVGP)):
        X = model.inducing_variable.Z
    else:
        X = model.data[0]
    N = X.shape[0]
    alpha = get_model_sufficient_statistics(model, get_L=False)
    sobol = []
    L_list = []
    for kernel in kernel_list:
        assert isinstance(kernel, KernelComponenent)
        if len(kernel.iComponent_list) == 0:
            continue  # skip constant term
        L = np.ones((N, N))
        n_order = len(kernel.kernels)
        for j in range(len(kernel.kernels)):
            if share_var_across_orders:
                if j < 1:
                    v = kernel.oak_kernel.variances[n_order].numpy()
                else:
                    v = 1
            else:
                v = kernel.kernels[j].base_kernel.variance.numpy()

            dim = kernel.kernels[j].active_dims[0]

            if isinstance(kernel.kernels[j], OrthogonalRBFKernel):

                if isinstance(kernel.kernels[j].base_kernel, gpflow.kernels.RBF) and (
                    not isinstance(kernel.kernels[j].measure, EmpiricalMeasure)
                    and (not isinstance(kernel.kernels[j].measure, MOGMeasure))
                ):
                    l = kernel.kernels[j].base_kernel.lengthscales.numpy()
                    L = L * compute_L(
                        X,
                        l,
                        v,
                        dim,
                        delta,
                        mu,
                    )

                elif isinstance(kernel.kernels[j].measure, EmpiricalMeasure):
                    L = (
                        v ** 2
                        * L
                        * compute_L_empirical_measure(
                            kernel.kernels[j].measure.location,
                            kernel.kernels[j].measure.weights,
                            kernel.kernels[j],
                            tf.reshape(X[:, dim], [-1, 1]),
                        )
                    )
                else:
                    raise NotImplementedError

            elif isinstance(kernel.kernels[j], OrthogonalBinary):
                p0 = kernel.kernels[j].p0
                L = L * compute_L_binary_kernel(X, p0, v, dim)

            elif isinstance(kernel.kernels[j], OrthogonalCategorical):
                p = kernel.kernels[j].p
                W = kernel.kernels[j].W
                kappa = kernel.kernels[j].kappa
                L = L * compute_L_categorical_kernel(X, W, kappa, p, v, dim)

            else:
                raise NotImplementedError
        L_list.append(L)
        mean_term = tf.tensordot(
            tf.tensordot(tf.transpose(alpha), L, axes=1), alpha, axes=1
        ).numpy()[0][0]
        sobol.append(mean_term)

    assert len(selected_dims_oak) == len(sobol)
    return selected_dims_oak, sobol


def compute_sobol(
    model: GPModel,
    kernel_list: list,
    delta: float,
    mu: float,
    alpha: np.ndarray,
    sparse_gp: bool = True,
):
    # compute Sobol in eq (40) of G.1 of paper
    if sparse_gp:
        X = model.inducing_variable.Z
    else:
        X = model.data[0]
    N = X.shape[0]
    sobol = []
    L_list = []
    for kernel in kernel_list:
        assert not isinstance(
            kernel, KernelComponenent
        ), "should use duvenaud sobol calculation code"
        if isinstance(kernel, gpflow.kernels.base.Product):  # exclude constant term
            L = np.ones((N, N))
            for j in range(len(kernel.kernels)):
                l = kernel.kernels[j].base_kernel.lengthscales.numpy()
                v = kernel.kernels[j].base_kernel.variance.numpy()
                dim = kernel.kernels[j].active_dims[0]
                L = L * compute_L(X, l, v, dim, delta, mu)
            L_list.append(L)
            sobol.append(
                tf.tensordot(
                    tf.tensordot(tf.transpose(alpha), L, axes=1), alpha, axes=1
                ).numpy()[0][0]
            )

        else:
            if type(kernel) != gpflow.kernels.statics.Constant and not isinstance(
                kernel, KernelComponenent
            ):
                l = kernel.base_kernel.lengthscales.numpy()
                v = kernel.base_kernel.variance.numpy()
                dim = kernel.active_dims[0]
                L = compute_L(X, l, v, dim, delta, mu)

                L_list.append(L)
                sobol.append(
                    tf.tensordot(
                        tf.tensordot(tf.transpose(alpha), L, axes=1), alpha, axes=1
                    ).numpy()[0][0]
                )

    return sobol


def get_prediction_component(
    m: gpflow.models.BayesianModel,
    alpha: tf.Tensor,
    X: np.ndarray = None,
    share_var_across_orders: Optional[bool] = True,
) -> list:
    """
    Return predictive mean for dataset 1 and 2
    :param m: GP model
    :param X: concatenation of data to make predictions: first half of X are from dataset 1,
              last half of X are from dataset 2. If it is None, then X is set to be the training data.
    :param alpha: statistics used to make predictions, e.g. K^{-1}y
    :param share_var_across_orders: whether to share the same variance across orders,
           if False, it uses original OrthogonalRBFKernel kernel \prod_i(1+k_i)
    :return:  prediction of each kernel component of two datasets (e.g., two different simulation runs), concatenated together
    """
    if X is None:
        X = m.data[0]
    selected_dims, _ = get_list_representation(m.kernel, num_dims=X.shape[1])
    tuple_of_indices = selected_dims[1:]
    out = []
    if isinstance(m, gpflow.models.GPR):
        X_conditioned = m.data[0]
    elif isinstance(m, (gpflow.models.SGPR, gpflow.models.SVGP)):
        X_conditioned = m.inducing_variable.Z

    for n in range(len(tuple_of_indices)):
        Kxx = tf.ones([X.shape[0], alpha.shape[0]], dtype=tf.dtypes.float64)
        num_interaction = len(tuple_of_indices[n])
        for ii in range(num_interaction):
            idx = tuple_of_indices[n][ii]
            Kxx *= m.kernel.kernels[idx].K(
                np.reshape(X[:, idx], (-1, 1)), X_conditioned[:, idx : idx + 1]
            )
        if share_var_across_orders:
            Kxx *= m.kernel.variances[num_interaction]

        predictive_component_mean = tf.matmul(Kxx, alpha)
        out.append(predictive_component_mean[:, 0])
    return out


def initialize_kmeans_with_binary(
    X: tf.Tensor,
    binary_index: list,
    continuous_index: Optional[list] = None,
    n_clusters: Optional[int] = 200,
):
    # K-means with combination of continuous and binary feature
    Z = np.zeros([n_clusters, X.shape[1]])

    for index in binary_index:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, index][:, None])
        Z[:, index] = kmeans.cluster_centers_.astype(int)[:, 0]

    if continuous_index is not None:
        kmeans_continuous = KMeans(n_clusters=n_clusters, random_state=0).fit(
            X[:, continuous_index]
        )
        Z[:, continuous_index] = kmeans_continuous.cluster_centers_

    return Z


def initialize_kmeans_with_categorical(
    X: tf.Tensor,
    binary_index: list,
    categorical_index: list,
    continuous_index: list,
    n_clusters: Optional[int] = 200,
):
    # K-means with combination of continuous and categorical feature
    Z = np.zeros([n_clusters, X.shape[1]])

    for index in binary_index + categorical_index:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, index][:, None])
        Z[:, index] = kmeans.cluster_centers_.astype(int)[:, 0]

    kmeans_continuous = KMeans(n_clusters=n_clusters, random_state=0).fit(
        X[:, continuous_index]
    )
    Z[:, continuous_index] = kmeans_continuous.cluster_centers_

    return Z
