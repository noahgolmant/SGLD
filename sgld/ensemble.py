""" Ensemble model that creates predictions using all loaded models """
import torch


class Ensemble(torch.nn.Module):
    """
    produces a single predictive distribution by averaging the softmax
    outputs for a list of pytorch models
    """
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

    def forward(self, inputs):
        # model(inputs) is BS x NUM_CLASSES
        predictions = [model(inputs) for model in self.models]
        # final predictive distribution is just the mean of the softmax output
        return sum(predictions) / len(self.models)
