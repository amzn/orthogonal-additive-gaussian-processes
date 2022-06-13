# Orthogonal Additive Gaussian Processes


This is the code repo for the paper Additive Gaussian Processes Revisited. 
Link to [paper latex] (TODO: add arXiv link after we publish the paper)


## Getting Started
### Installation
Clone the repository (TODO: add github repo) and install the package with `python setup.py install` and `pip install -e .`. 
The main dependency is `gpflow` and we relied on `gpflow == 2.2.1`, where in particular implements the posteriors module.

### Tests
Run `pytest` to run the tests in the `tests` folder.

### Key Components

- Kernels:
	- `ortho_binary_kernel.py` implements the constrained binary kernel 

	- `ortho_categorical_kernel.py` implements the constrained coregional kernel for categorical variables

	- `ortho_rbf_kernel.py` implements the constrained squared exponential (SE) kernel for continuous variables
	
	- `oak_kernel.py` multiples and adds kernels over feature dimensions using Newton Girard method

- Measures:
	- `input_measures.py` implements Uniform measure, (mixture of) Gaussian measure, empirical measure for input distributions


- Normalising Flow:
	- `normalising_flow.py` implements normalising flows to transform input densities into Gaussian random variables 


- Model API:
	- `model_utils.py` is the model API for model inference, prediction and plotting, and Sobol calculations

- Utilities:
	- `utils.py` contains utility functions 
	- `plotting_utils.py` contains utility functions for plotting


## Usage

**Data**

UCI benchmark data are saved in the `./data` directory. They are obtained from https://github.com/duvenaud/additive-gps/blob/master/data/. Run `./data/download_data.py` to download all the datasets. 

**Examples**

Example tutorials and scripts are in the `./example` directory.

*UCI:*

* Contains training scripts for UCI regression and classification
benchmark datasets. See `./examples/uci/README_UCI.md` for details. 

