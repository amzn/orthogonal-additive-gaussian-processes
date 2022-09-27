# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup
# -

setup(
    name="oak",
    version="0.0.1",
    packages=find_packages(include=['oak/oak', 'oak.*']),
    install_requires=[
        "gpflow",
        "tensorflow",
        "tensorflow_probability",
        "numpy",
        "scipy",
        "scikit-learn",
        "scikit-learn-extra",
        "matplotlib",
        "tikzplotlib",
    ],
)
