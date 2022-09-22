# SHGP - Sparse Hierarchical Gaussian Process
A sparse implementation of a heirarchical Gaussian processes in gpflow. On the jax branch we are planning to produce a jax implementation.

**Please note that there is an active issue where the compilation of models with a large number of time series (realisations) create memory problems. We're working on this**

### Setup 
Create conda environment
```shell
conda env create -f environment.yml
conda activate shgp-env
```
Install package:
```shell
python setup.py install .
```

### Examples
There is a basic data example at examples/basic_example.ipynb and a climate modelling based example at examples/climate_modelling_example_1D.ipynb
