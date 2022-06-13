# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Union
import gpflow
import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPR, SGPR, GPModel
from gpflow.models.training_mixins import RegressionData
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tensorflow_probability import distributions as tfd
from oak import plotting_utils
from oak.input_measures import MOGMeasure
from oak.normalising_flow import Normalizer
from oak.oak_kernel import OAKKernel, get_list_representation
from oak.plotting_utils import FigureDescription, save_fig_list
from oak.utils import compute_sobol_oak, initialize_kmeans_with_categorical
# -

f64 = gpflow.utilities.to_default_float


def get_kmeans_centers(X: np.ndarray, K: int = 500) -> np.ndarray:
    """
    :param X: N * D input array
    :param K: number of clusters
    :return: K-means clustering of input X
    """
    np.random.seed(44)
    tf.random.set_seed(44)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    Z = kmeans.cluster_centers_
    return Z


def save_model(
    model: GPModel,
    filename: Path,
) -> None:
    """
    :param model: GPflow model parameters to save
    :param filename: location to save the model to
    :return save model parameters to a local directory
    """
    if isinstance(model, gpflow.models.SVGP):
        hyperparams = [
            model.parameters[i].numpy() for i in range(len(model.parameters))
        ]
    else:
        hyperparams = [
            model.trainable_parameters[i].numpy()
            for i in range(len(model.trainable_parameters))
        ]

    os.makedirs(filename.parents[0], exist_ok=True)
    np.savez(filename, hyperparams=hyperparams)


def load_model(
    model: GPModel,
    filename: Path,
    load_all_parameters=False,
) -> None:
    """
    :param model: GPflow model parameters to load
    :param filename: location to load the model from
    :param load_all_parameters: whether to load all parameters or only trainable parameters
    :return load model parameters from a local directory
    """
    # We need allow_pickle=True because model parameters include objects (e.g. InducingPoints)
    model_params = np.load(str(filename), allow_pickle=True)["hyperparams"]

    if load_all_parameters:
        for i in range(len(model.parameters)):
            model.parameters[i].assign(model_params[i])
    else:
        for i in range(len(model.trainable_parameters)):
            print(model_params[i], model.trainable_parameters[i])
            model.trainable_parameters[i].assign(model_params[i])


def create_model_oak(
    data: RegressionData,
    max_interaction_depth: int = 2,
    constrain_orthogonal: bool = True,
    inducing_pts: np.ndarray = None,
    optimise=False,
    zfixed=True,
    p0=None,
    p=None,
    lengthscale_bounds=None,
    empirical_locations: Optional[List[float]] = None,
    empirical_weights: Optional[List[float]] = None,
    use_sparsity_prior: bool = True,
    gmm_measures: Optional[List[MOGMeasure]] = None,
    share_var_across_orders: Optional[bool] = True,
) -> GPModel:
    """
    :param num_dims: number of dimensions of inputs
    :param max_interaction_depth: maximum order of interactions
    :param constrain_orthogonal: whether to use the orthogonal version of the kernel
    :param inducing_pts: inducing points, if None, it uses K-means centers
    :param optimise: whether to optimise the hyper parameters of the model
    :param zfixed: whether to fix or learn the inducing points
    :param p0: list of probability measures for binary kernels, set to None if it is not binary
    :param p: list of probability measures for categorical kernels, set to None if it is not categorical
    :param lengthscale_bounds: bounds of the lengthscale parameters
    :param empirical_locations: list of locations of empirical measure, set to None if not using the empirical measure
    :param empirical_weights: list of weights of empirical measure, set to None if not using the empirical measure
    :param use_sparsity_prior: whether to use sparse prior on the kernel variance parameters
    :param gmm_measures: list of Gaussian mixture measures
    :param share_var_across_orders: whether to use the same variance parameter across interaction order
    :return: a GP model with OAK kernel
    """
    num_dims = data[0].shape[1]

    # create oak kernel
    if p0 is None:
        p0 = [None] * num_dims
    if p is None:
        p = [None] * num_dims
    base_kernels = [None] * num_dims
    for dim in range(num_dims):
        if (p0[dim] is None) and (p[dim] is None):
            base_kernels[dim] = gpflow.kernels.RBF

    k = OAKKernel(
        base_kernels,
        num_dims=num_dims,
        max_interaction_depth=max_interaction_depth,
        constrain_orthogonal=constrain_orthogonal,
        p0=p0,
        p=p,
        lengthscale_bounds=lengthscale_bounds,
        empirical_locations=empirical_locations,
        empirical_weights=empirical_weights,
        gmm_measures=gmm_measures,
        share_var_across_orders=share_var_across_orders,
    )

    if inducing_pts is not None:
        model = SGPR(
            data,
            mean_function=None,
            kernel=k,
            inducing_variable=InducingPoints(inducing_pts),
        )
        if zfixed:
            set_trainable(model.inducing_variable, False)
    else:
        model = GPR(data, mean_function=None, kernel=k)
    # set priors for variance
    if use_sparsity_prior:
        print("Using sparsity prior")
        if share_var_across_orders:
            for p in model.kernel.variances:
                p.prior = tfd.Gamma(f64(1.0), f64(0.2))
    # Initialise likelihood variance to small value to avoid finding all-noise explanation minima
    model.likelihood.variance.assign(0.01)
    if optimise:
        t_start = time.time()
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            model.training_loss_closure(), model.trainable_variables, method="BFGS"
        )
        gpflow.utilities.print_summary(model, fmt="notebook")
        print(f"Training took {time.time() - t_start:.1f} seconds.")
    return model


def apply_normalise_flow(X: tf.Tensor, input_flows: List[Normalizer]) -> tf.Tensor:
    """
    :param X: input of which the normalising flow is to be applied
    :param input_flows: list of normalising flows to apply to each feature dimension
    :return: inputs after transformations of the flow
    """
    X_scaled = np.zeros((X.shape))
    for ii in range(X.shape[1]):
        if input_flows[ii] is None:
            X_scaled[:, ii] = X[:, ii]
        else:
            X_scaled[:, ii] = input_flows[ii].bijector(X[:, ii])
    return X_scaled


class oak_model:
    def __init__(
        self,
        max_interaction_depth=2,
        num_inducing=200,
        lengthscale_bounds=[1e-3, 1e3],
        binary_feature: Optional[List[int]] = None,
        categorical_feature: Optional[List[int]] = None,
        empirical_measure: Optional[List[int]] = None,
        use_sparsity_prior: bool = True,
        gmm_measure: Optional[List[int]] = None,
        sparse: bool = False,
        use_normalising_flow: bool = True,
        share_var_across_orders: bool = True,
    ):
        """
        :param max_interaction_depth: maximum number of interaction terms to consider
        :param num_inducing: number of inducing points
        :param lengthscale_bounds: bounds for lengthscale parameters
        :param binary_feature: list of indices for binary features
        :param categorical_feature: list of indices for categorical features
        :param empirical_measure: list of indices using empirical measures, if using Gaussian measure, this is set to None
        :param use_sparsity_prior: use sparsity prior on kernel variances
        :param gmm_measure: use gaussian mixture model. If index is 0 it will use a Gaussian measure, otherwise
        :param sparse: Boolean to indicate whether to use sparse GP with inducing points. Defaults to False.
        :param use_normalising_flow: whether to use normalising flow, if not, continuous features are standardised
        :param share_var_across_orders: whether to share the same variance across orders,
           if False, it uses kernel of the form \prod_i(1+k_i) in Duvenaud (2011).
        :return: OAK model class with model fitting, prediction, attribution and plotting utils.
        """
        self.max_interaction_depth = max_interaction_depth
        self.num_inducing = num_inducing
        self.lengthscale_bounds = lengthscale_bounds
        self.binary_feature = binary_feature
        self.categorical_feature = categorical_feature
        self.use_sparsity_prior = use_sparsity_prior

        # state filled in during fit call
        self.input_flows = None
        self.scaler_y = None
        self.Y_scaled = None
        self.X_scaled = None
        self.alpha = None
        self.continuous_index = None
        self.binary_index = None
        self.categorical_index = None
        self.empirical_measure = empirical_measure
        self.empirical_locations = None
        self.empirical_weights = None
        self.gmm_measure = gmm_measure
        self.estimated_gmm_measures = None  # sklearn GMM estimates
        self.sparse = sparse
        self.use_normalising_flow = use_normalising_flow
        self.share_var_across_orders = share_var_across_orders

    def fit(
        self,
        X: tf.Tensor,
        Y: tf.Tensor,
        optimise: bool = True,
        initialise_inducing_points: bool = True,
    ):
        """
        :param X, Y data to fit the model on
        :param optimise: whether to optimise the model
        :param initialise_inducing_points: whether to initialise inducing points with K-means
        """
        self.xmin, self.xmax = X.min(0), X.max(0)
        self.num_dims = X.shape[1]

        (
            self.continuous_index,
            self.binary_index,
            self.categorical_index,
            p0,
            p,
        ) = _calculate_features(
            X,
            categorical_feature=self.categorical_feature,
            binary_feature=self.binary_feature,
        )
        # discrete_input_set = set(self.binary_index).union(set(self.categorical_index))
        if self.empirical_measure is not None:
            if not set(self.empirical_measure).issubset(self.continuous_index):
                raise ValueError(
                    f"Empirical measure={self.empirical_measure} should only be used on non-binary/categorical inputs {self.continuous_index}"
                )
        if self.gmm_measure is not None:
            if len(self.gmm_measure) != self.num_dims:
                return ValueError(
                    f"Must specify number of components for each inputs dimension 1..{X.shape[0]}"
                )
            idx_gmm = np.flatnonzero(self.gmm_measure)
            if not set(idx_gmm).issubset(self.continuous_index):
                raise ValueError(
                    f"GMM measure on inputs {idx_gmm} should only be used on continuous inputs {self.continuous_index}"
                )

        # Measure
        self.estimated_gmm_measures = [None] * self.num_dims
        if self.gmm_measure is not None:
            for i_dim in np.flatnonzero(self.gmm_measure):
                K_for_input_i = self.gmm_measure[i_dim]
                self.estimated_gmm_measures[i_dim] = estimate_one_dim_gmm(
                    K=K_for_input_i, X=X[:, i_dim]
                )

        self.empirical_locations = [None] * self.num_dims
        self.empirical_weights = [None] * self.num_dims

        # scaling
        self.input_flows = [None] * self.num_dims
        for i in self.continuous_index:
            if (self.empirical_measure is not None) and (i in self.empirical_measure):
                continue  # skip
            if self.estimated_gmm_measures[i] is not None:
                continue
            d = X[:, i]

            if self.use_normalising_flow:
                n = Normalizer(d)
                opt = gpflow.optimizers.Scipy()
                opt.minimize(n.KL_objective, n.trainable_variables)
                self.input_flows[i] = n

        self.alpha = None
        self.scaler_y = preprocessing.StandardScaler().fit(Y)
        self.Y_scaled = self.scaler_y.transform(Y)
        # standardize features with empirical measure to avoid cholesky decomposition error
        if self.empirical_measure is not None:
            self.scaler_X_empirical = preprocessing.StandardScaler().fit(
                X[:, self.empirical_measure]
            )
        if not self.use_normalising_flow:
            self.scaler_X_continuous = preprocessing.StandardScaler().fit(
                X[:, self.continuous_index]
            )
        self.X_scaled = self._transform_x(X)

        # calculate empirical location and weights after applying scaling X
        if self.empirical_measure is not None:
            for ii in self.empirical_measure:
                self.empirical_locations[ii], self.empirical_weights[ii] = np.unique(
                    self.X_scaled[:, ii], return_counts=True
                )
                self.empirical_weights[ii] = (
                    self.empirical_weights[ii] / self.empirical_weights[ii].sum()
                ).reshape(-1, 1)
                self.empirical_locations[ii] = self.empirical_locations[ii].reshape(
                    -1, 1
                )

        assert np.allclose(
            self.X_scaled[:, self.binary_index], X[:, self.binary_index]
        ), "Flow applied to binary inputs"
        assert np.allclose(
            self.X_scaled[:, self.categorical_index], X[:, self.categorical_index]
        ), "Flow applied to categorical inputs"
        if self.gmm_measure is not None:
            assert np.allclose(
                self.X_scaled[:, np.flatnonzero(self.gmm_measure)],
                X[:, np.flatnonzero(self.gmm_measure)],
            ), "Flow applied to GMM measure inputs"
        if self.empirical_measure is not None:
            assert np.allclose(
                np.reshape(
                    np.concatenate(
                        [
                            self._get_x_inverse_transformer(i)(self.X_scaled[:, i])
                            for i in self.empirical_measure
                        ]
                    ),
                    X[:, self.empirical_measure].shape,
                    order="F",
                ),
                X[:, self.empirical_measure],
            ), "Flow applied to empirical measure inputs"

        Z = None
        # using sparse GP when size of data > 1000
        if X.shape[0] > 1000 or self.sparse:
            X_inducing = self.X_scaled

            if initialise_inducing_points:
                if (p0 is None) and (p is None):
                    print("all features are continuous")
                    kmeans = KMeans(n_clusters=self.num_inducing, random_state=0).fit(
                        X_inducing
                    )
                    Z = kmeans.cluster_centers_
                else:
                    Z = initialize_kmeans_with_categorical(
                        X_inducing,
                        binary_index=self.binary_index,
                        categorical_index=self.categorical_index,
                        continuous_index=self.continuous_index,
                        n_clusters=self.num_inducing,
                    )
            else:
                Z = X_inducing[: self.num_inducing, :]

        self.m = create_model_oak(
            (self.X_scaled, self.Y_scaled),
            max_interaction_depth=self.max_interaction_depth,
            inducing_pts=Z,
            optimise=optimise,
            p0=p0,
            p=p,
            lengthscale_bounds=self.lengthscale_bounds,
            use_sparsity_prior=self.use_sparsity_prior,
            empirical_locations=self.empirical_locations,
            empirical_weights=self.empirical_weights,
            gmm_measures=self.estimated_gmm_measures,
            share_var_across_orders=self.share_var_across_orders,
        )

    def optimise(
        self,
        compile: bool = True,
    ):

        print("Model prior to optimisation")
        gpflow.utilities.print_summary(self.m, fmt="notebook")
        self.alpha = None
        t_start = time.time()
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            self.m.training_loss_closure(),
            self.m.trainable_variables,
            method="BFGS",
            compile=compile,
        )
        gpflow.utilities.print_summary(self.m, fmt="notebook")
        print(f"Training took {time.time() - t_start:.1f} seconds.")

    def predict(self, X: tf.Tensor, clip=False) -> tf.Tensor:
        """
        :param X: inputs to predict the response on
        :param clip: whether to slip X between x_min and x_max along each dimension
        :return: predicted response on input X
        """
        if clip:
            X_scaled = self._transform_x(np.clip(X, self.xmin, self.xmax))
        else:
            X_scaled = self._transform_x(X)
        try:
            y_pred = self.m.predict_f(X_scaled)[0].numpy()
            return self.scaler_y.inverse_transform(y_pred)[:, 0]
        except ValueError:
            print("test X is outside the range of training input, try clipping X.")

    def get_loglik(self, X: tf.Tensor, y: tf.Tensor, clip=False) -> tf.Tensor:
        """
        :param X,y: inputs and output
        :param clip: whether to slip X between x_min and x_max along each dimension
        :return log likelihood on (X,y)
        """
        if clip:
            X_scaled = self._transform_x(np.clip(X, self.xmin, self.xmax))
        else:
            X_scaled = self._transform_x(X)

        return (
            self.m.predict_log_density((X_scaled, self.scaler_y.transform(y)))
            .numpy()
            .mean()
        )

    def _transform_x(self, X: tf.Tensor) -> tf.Tensor:
        """
        :param X: input to do transformation on
        :return: transformation for continuous features: normalising flow with Gaussian measure or standardization with empirical measure
        """
        X = apply_normalise_flow(X, self.input_flows)
        if self.empirical_measure is not None:
            X[:, self.empirical_measure] = self.scaler_X_empirical.transform(
                X[:, self.empirical_measure]
            )
        if not self.use_normalising_flow:
            X[:, self.continuous_index] = self.scaler_X_continuous.transform(
                X[:, self.continuous_index]
            )
        return X

    def _get_x_inverse_transformer(
        self, i: int
    ) -> Optional[Union[Normalizer, Callable[[tf.Tensor], tf.Tensor]]]:
        """
        :param i: index of feature i
        :return: inverse transformation for continuous feature i
        """
        assert i in self.continuous_index

        if self.empirical_measure is not None and i in self.empirical_measure:
            continuous_i = self.empirical_measure.index(i)
            mean_i, std_i = self.scaler_X_empirical.mean_[continuous_i], np.sqrt(
                self.scaler_X_empirical.var_[continuous_i]
            )
            transformer_x = lambda x: x * std_i + mean_i
        elif self.gmm_measure is not None and i in self.gmm_measure:
            transformer_x = None
        else:
            transformer_x = self.input_flows[i].bijector.inverse
        return transformer_x

    def get_sobol(self, likelihood_variance=False):
        """
        :param likelihood_variance: whether to include likelihood noise in Sobol calculation
        :return: normalised Sobol indices for each additive term in the model
        """
        num_dims = self.num_dims

        delta = 1
        mu = 0
        selected_dims, _ = get_list_representation(self.m.kernel, num_dims=num_dims)
        tuple_of_indices = selected_dims[1:]
        model_indices, sobols = compute_sobol_oak(
            self.m,
            delta,
            mu,
            share_var_across_orders=self.share_var_across_orders,
        )
        if likelihood_variance:
            normalised_sobols = sobols / (
                np.sum(sobols) + self.m.likelihood.variance.numpy()
            )
        else:
            normalised_sobols = sobols / np.sum(sobols)
        self.normalised_sobols = normalised_sobols
        self.tuple_of_indices = tuple_of_indices
        return normalised_sobols

    def plot(
        self,
        transformer_y=None,
        X_columns=None,
        X_lists=None,
        top_n=None,
        likelihood_variance=False,
        semilogy=True,
        save_fig: Optional[str] = None,
        tikz_path: Optional[str] = None,
        ylim: Optional[List[float]] = None,
        quantile_range: Optional[List[float]] = None,
        log_axis: Optional[List[bool]] = [False, False],
        grid_range: Optional[List[np.ndarray]] = None,
        log_bin: Optional[List[bool]] = None,
        num_bin: Optional[int] = 100,
    ):
        """
        :param transformer_y: tranformation of the target (e.g. log), we are plotting the median and quantiles after log-transformation
        :param X_columns: list of feature names
        :param X_list: list of features from data 1 and data 2, if None, then training features will be plotted on the histogram
        :param top_n: plot top n effects based on sobol indices
        :param likelihood_variance: Whether to add the likelihood variance or not to the total Sobol
        :param save_fig: save the figure saved in the directory
        :param tikz_path: save latex for figures in the directory
        :param ylim: list of limits on the y-axis for each feature
        :param quantile_range: list of quantile range of each feature to plot. If None, use the whole range
        :param log_axis: Boolean indicating whether to log x-axis and y-axis for the contour plot
        :param grid_range: list of ranges to plot functions on the contour plot for each feature, if None, use linspace of the feature ranges
        :param log_bin: list of Booleans indicating whether to log bins for histograms for each feature
        :param num_bin: number of bins for histogram
        :return: plotting of individual effects
        """
        if X_columns is None:
            X_columns = ["feature %d" % i for i in range(self.num_dims)]

        if X_lists is None:
            X_lists = [None for i in range(len(X_columns))]

        if grid_range is None:
            grid_range = [None for i in range(len(X_columns))]

        if ylim is None:
            ylim = [None for i in range(len(X_columns))]

        if quantile_range is None:
            quantile_range = [None for i in range(len(X_columns))]

        if log_bin is None:
            log_bin = [False for i in range(len(X_columns))]

        num_dims = self.num_dims
        selected_dims, _ = get_list_representation(self.m.kernel, num_dims=num_dims)
        tuple_of_indices = selected_dims[1:]

        self.get_sobol(likelihood_variance=likelihood_variance)
        order = np.argsort(self.normalised_sobols)[::-1]
        fig_list: List[FigureDescription] = []
        if top_n is None:
            top_n = len(order)
        for n in order[: min(top_n, len(order))]:
            if len(tuple_of_indices[n]) == 1:
                i = tuple_of_indices[n][0]
                if i in self.continuous_index:
                    fig_list.append(
                        plotting_utils.plot_single_effect(
                            m=self.m,
                            i=i,
                            covariate_name=X_columns[i],
                            title=f"{X_columns[i]} (R={self.normalised_sobols[n]:.3f})",
                            x_transform=self._get_x_inverse_transformer(i),
                            y_transform=transformer_y,
                            semilogy=semilogy,
                            plot_corrected_data=False,
                            plot_raw_data=False,
                            X_list=X_lists[i],
                            tikz_path=tikz_path,
                            ylim=ylim[i],
                            quantile_range=quantile_range[i],
                            log_bin=log_bin[i],
                            num_bin=num_bin,
                        )
                    )

                elif i in self.binary_index:
                    fig_list.append(
                        plotting_utils.plot_single_effect_binary(
                            self.m,
                            i,
                            ["0", "1"],
                            title=f"{X_columns[i]} (R={self.normalised_sobols[n]:.3f})",
                            y_transform=transformer_y,
                            semilogy=semilogy,
                            tikz_path=tikz_path,
                        )
                    )
                else:
                    fig_list.append(
                        plotting_utils.plot_single_effect_categorical(
                            self.m,
                            i,
                            [str(i) for i in range(self.m.kernel.kernels[i].num_cat)],
                            title=f"{X_columns[i]} (R={self.normalised_sobols[n]:.3f})",
                            y_transform=transformer_y,
                            semilogy=semilogy,
                            tikz_path=tikz_path,
                        )
                    )

            elif len(tuple_of_indices[n]) == 2:
                i = tuple_of_indices[n][0]
                j = tuple_of_indices[n][1]
                if i in self.continuous_index and j in self.continuous_index:
                    fig_list.append(
                        plotting_utils.plot_second_order(
                            self.m,
                            i,
                            j,
                            [X_columns[i], X_columns[j]],
                            [
                                self._get_x_inverse_transformer(i),
                                self._get_x_inverse_transformer(j),
                            ],
                            transformer_y,
                            title=X_columns[i]
                            + "&"
                            + X_columns[j]
                            + f" (R={self.normalised_sobols[n]:.3f})",
                            tikz_path=tikz_path,
                            quantile_range=[quantile_range[i], quantile_range[j]],
                            log_axis=log_axis,
                            xx=grid_range[i],
                            yy=grid_range[j],
                            num_bin=num_bin,
                        )
                    )

                elif i in self.continuous_index and j in self.binary_index:
                    fig_list.append(
                        plotting_utils.plot_second_order_binary(
                            self.m,
                            i,
                            j,
                            ["0", "1"],
                            [X_columns[i], X_columns[j]],
                            x_transforms=[self._get_x_inverse_transformer(i)],
                            y_transform=transformer_y,
                            title=f"{X_columns[i]} (R={self.normalised_sobols[n]:.3f})",
                            tikz_path=tikz_path,
                        )
                    )

                elif i in self.binary_index and j in self.continuous_index:
                    fig_list.append(
                        plotting_utils.plot_second_order_binary(
                            self.m,
                            j,
                            i,
                            ["0", "1"],
                            [X_columns[j], X_columns[i]],
                            x_transforms=[self._get_x_inverse_transformer(j)],
                            y_transform=transformer_y,
                            title=X_columns[i]
                            + "&"
                            + X_columns[j]
                            + f" (R={self.normalised_sobols[n]:.3f})",
                            tikz_path=tikz_path,
                        )
                    )

            else:
                raise NotImplementedError

        if save_fig is not None:
            save_fig_list(fig_list=fig_list, dirname=Path(save_fig))


def _calculate_features(
    X: tf.Tensor, categorical_feature: List[int], binary_feature: List[int]
):
    """
    Calculate features index set
    :param X: input data
    :param categorical_feature: index of categorical features
    :param binary_feature: index of binary features
    :return:
        continuous_index, binary_index, categorical_index: list of indices for type of feature
        p0: list of probability measure for binary kernels, for continuous/categorical kernel, it is set to None
        p: list of probability measure for categorical kernels, for continuous/binary kernel, it is set to None
    """
    if binary_feature is None and categorical_feature is None:
        # all features are continuous
        p0 = None
        p = None
        continuous_index = list(range(X.shape[1]))
        binary_index = []
        categorical_index = []
    else:
        if binary_feature is not None and categorical_feature is not None:
            overlapping_set = set(binary_feature).intersection(categorical_feature)
            if len(overlapping_set) > 0:
                raise ValueError(f"Overlapping feature set {overlapping_set}")
        binary_index, categorical_index, continuous_index, p0, p = [], [], [], [], []
        for j in range(X.shape[1]):
            if binary_feature is not None and j in binary_feature:
                p0.append(1 - X[:, j].mean())
                p.append(None)
                binary_index.append(j)
            elif categorical_feature is not None and j in categorical_feature:
                p0.append(None)
                prob = []
                for jj in np.unique(X[:, j]):
                    prob.append(len(np.where(X[:, j] == jj)[0]) / len(X[:, j]))
                p.append(np.reshape(prob, (-1, 1)))
                assert np.abs(p[-1].sum() - 1) < 1e-6
                categorical_index.append(j)
            else:
                p.append(None)
                p0.append(None)
                continuous_index.append(j)
    print("indices of binary feature ", binary_index)
    print("indices of continuous feature ", continuous_index)
    print("indices of categorical feature ", categorical_index)

    return continuous_index, binary_index, categorical_index, p0, p


def estimate_one_dim_gmm(K: int, X: np.ndarray) -> MOGMeasure:
    """
    :param K: number of mixtures
    :param X: input data
    :return: estimated Gaussian mixture model on the data X
    """
    tf.debugging.assert_shapes([(X, ("N",))])
    assert K > 0
    gm = GaussianMixture(
        n_components=K, random_state=0, covariance_type="spherical"
    ).fit(X.reshape(-1, 1))
    assert np.allclose(gm.weights_.sum(), 1.0)
    assert gm.means_.shape == (K, 1)
    assert gm.covariances_.shape == (K,)
    assert gm.weights_.shape == (K,)
    return MOGMeasure(
        weights=gm.weights_, means=gm.means_.reshape(-1), variances=gm.covariances_
    )
