# SGLD
Stochastic Gradient Langevin Dynamics for Bayesian learning

To train the models for all noise scale configurations, use
`python train.py --config config.yaml sgld`

To train a single model, do
`python train.py <args> sgld`

To perform some experiment analysis, do
`python analysis.py --mode <experiment> <otherargs> analysis`

You can then do the plotting and stuff in any of the notebook in `notebooks/`
