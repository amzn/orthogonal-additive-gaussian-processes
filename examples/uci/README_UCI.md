<!-- #region -->
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0

# UCI experiments

## Running scripts 
* To run the experiments on regression dataset, run 
```
python examples/uci/uci_regression_train.py --dataset_name=NAME
```
* on classification dataset, run
```
python examples/uci/uci_classification_train.py --dataset_name=NAME
```
where ```NAME``` is the name of the dataset (regression data: ```autoMPG, Housing, concrete, pumadyn``` and classification data: ```breast, pima, sonar, ionosphere, liver, heart```. The two scripts save the model and metrics in the ```example/uci/outputs/``` folder.

To visualise the functional decomposition, run 
```
python examples/uci/uci_plotting.py --dataset_name=NAME
```
This will save the plots in the ```example/uci/outputs/NAME/decomposition``` folder. 
For illustration, we have run the scripts for the autoMPG and breast datasets, with results saved in the above output folder. 


## Example notebook
We provide one example notebook on the AutoMPG UCI regression problem, in ```example/uci/example_autompg.ipynb```.
<!-- #endregion -->

```python

```
