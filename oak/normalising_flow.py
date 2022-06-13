# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from scipy import stats
# -

tfb = tfp.bijectors


def make_sinharcsinh():
    return tfb.SinhArcsinh(
        skewness=gpflow.Parameter(0.0),
        tailweight=gpflow.Parameter(1.0, transform=tfb.Exp()),
    )


def make_standardizer(x):
    return [
        tfb.Scale(gpflow.Parameter(1.0 / np.std(x), transform=tfb.Exp())),
        tfb.Shift(gpflow.Parameter(-np.mean(x))),
    ]


class Normalizer(gpflow.base.Module):
    """
    :param x: input to transform
    :param log: whether to log x first before applying flows of transformations
    :return: flows of transformations to match x to standard Gaussian
    """

    def __init__(
        self,
        x,
        log=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x = x

        if log:
            offset = np.min(x) - 1.0
            self.bijector = tfb.Chain(
                [make_sinharcsinh() for _ in range(1)]
                + make_standardizer(np.log(x - offset))
                + [tfb.Log(), tfb.Shift(-offset)]
            )
        else:
            self.bijector = tfb.Chain(
                [make_sinharcsinh() for _ in range(1)] + make_standardizer(x)
            )

    def plot(self, title='Normalising Flow'):
        f = plt.figure()
        ax = f.add_axes([0.3, 0.3, 0.65, 0.65])
        x = self.x
        y = self.bijector(x).numpy()
        ax.plot(x, y, "k.", label="Gaussian")
        ax.legend()
        
        ax_x = f.add_axes([0.3, 0.05, 0.65, 0.25], sharex=ax)
        ax_x.hist(x, bins=20)
        ax_y = f.add_axes([0.05, 0.3, 0.25, 0.65], sharey=ax)
        ax_y.hist(y, bins=20, orientation="horizontal")
        ax_y.set_xlim(ax_y.get_xlim()[::-1])
        plt.title(title)
        

    def KL_objective(self):
        return 0.5 * tf.reduce_mean(
            tf.square(self.bijector(self.x))
        ) - tf.reduce_mean(
            self.bijector.forward_log_det_jacobian(self.x, event_ndims=0)
        )

    def kstest(self):
        # Kolmogorov-Smirnov test for normality of transformed data
        s, pvalue = stats.kstest(self.bijector(self.x).numpy()[:,0], "norm")
        print("KS test statistic is %.3f, p-value is %.8f" % (s, pvalue))
        return s, pvalue
