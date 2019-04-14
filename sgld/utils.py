""" utility functions for the ensemble / trial analysis """
import numpy as np
import torch


def entropy(predictions):
    """ compute shannon entropy of the categorical prediction """
    predictions = predictions / torch.sum(predictions, axis=1)
    return -torch.sum(predictions * torch.log(predictions) / np.log(2), axis=1)
