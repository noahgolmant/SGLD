# SGLD
Stochastic Gradient Langevin Dynamics for Bayesian learning

[*Final Project Writeup*](EECS126_Final_Project_SGLD.pdf)

This respository contains code to reproduce and analyze the results of the paper ["Bayesian Learning via Stochastic Gradient Langevin Dynamics"](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf). We evaluated the performance of SGLD as an ensembling technique, performed visualizations of the class activations for the model samples from the posterior, and estimated the model uncertainty by measuring the Shannon entropy of the model predictions in different setups.

## Running SGLD

This repository uses the [skeletor](https://github.com/noahgolmant/skeletor) framework for experiment orchestration and metric analysis.

To train the models for all noise scale configurations, use:
`python -m sgld.train --config config.yaml sgld`

To run an experiment, use:
`python -m sgld.analysis --mode <experiment> <experimentargs> analysis`

The code to reproduce the plots from the writeup can be found in `notebooks/`.
