# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import urllib.request

# download UCI datasets from https://github.com/duvenaud/additive-gps/ and save to ./data directory


data_path_prefix = os.path.abspath(os.path.join(os.path.dirname(__file__), "./")) + "/"

regression_filenames = [
    "autompg.mat",
    "housing.mat",
    "r_concrete_1030.mat",
    "pumadyn8nh.mat",
]

classification_filenames = [
    "breast.mat",
    "pima.mat",
    "sonar.mat",
    "ionosphere.mat",
    "r_liver.mat",
    "r_heart.mat",
]


for filename in regression_filenames + classification_filenames:
    if not os.path.isfile(filename):
        if filename == "autompg.mat":
            url = f'https://github.com/duvenaud/additive-gps/raw/master/data/regression/autompg/{filename}'
        else:
            if filename in regression_filenames:
                url = f'https://github.com/duvenaud/additive-gps/raw/master/data/regression/{filename}'
            else:
                url = f'https://github.com/duvenaud/additive-gps/raw/master/data/classification/{filename}'
        print(f"Downloading {filename}")
        urllib.request.urlretrieve(url, data_path_prefix + filename)

