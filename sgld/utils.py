""" utility functions for the ensemble / trial analysis """
import numpy as np
import torch


def entropy(predictions):
    """ compute shannon entropy of the categorical prediction """
    predictions = predictions / torch.sum(predictions, dim=1)
    log_probs = torch.log(predictions)
    # prevent nans with p(x) = 0
    log_probs[log_probs == float("-Inf")] = 0.0
    return -torch.sum(predictions * log_probs / np.log(2), dim=1)


if __name__ == '__main__':
    x = np.array([[.5, .5]])
    x = torch.from_numpy(x)
    assert entropy(x).item() == 1.
    x = np.array([[1., 0]])
    x = torch.from_numpy(x)
    assert entropy(x).item() == 0.
