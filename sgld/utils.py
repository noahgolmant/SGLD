""" utility functions for the ensemble / trial analysis """
import numpy as np
from skeletor.datasets import build_dataset
import torch


def entropy(predictions):
    """ compute shannon entropy of the categorical prediction """
    predictions = predictions / torch.sum(predictions, dim=1)
    log_probs = torch.log(predictions)
    # prevent nans with p(x) = 0
    log_probs[log_probs == float("-Inf")] = 0.0
    return -torch.sum(predictions * log_probs / np.log(2), dim=1)


def build_single_class_dataset(name, class_ind=0, **dataset_params):
    """
    wrapper for the base skeletor dataset loader `build_dataset`
    this will take in the same arguments, but the loader will only iterate
    over examples of the given class

    I'm just going to overwrite standard cifar loading data for now
    """
    assert name == 'cifar10', "we only have this one for now lol"
    trainloader, testloader = build_dataset(name, **dataset_params)

    def _filter(loader, mode='train'):
        dataset = loader.dataset
        data_attr = mode + '_data'  # e.g. train imgs in dataset.train_data
        label_attr = mode + '_labels'
        assert hasattr(dataset, data_attr) and hasattr(dataset, label_attr),\
            "cifar10 dataset obj has data array"
        data = getattr(dataset, data_attr)
        targets = np.array(getattr(dataset, label_attr))
        class_inds = np.where(targets == int(class_ind))
        data, targets = data[class_inds], targets[class_inds]
        setattr(dataset, data_attr, data)
        setattr(dataset, label_attr, targets)
        return loader
    return _filter(trainloader, mode='train'), _filter(testloader, mode='test')


def _test_shannon_entropy():
    x = np.array([[.5, .5]])
    x = torch.from_numpy(x)
    assert entropy(x).item() == 1.
    x = np.array([[1., 0]])
    x = torch.from_numpy(x)
    assert entropy(x).item() == 0.


def _test_filter_dataset():
    train, test = build_single_class_dataset('cifar10', class_ind=0,
                                             dataroot='../data',
                                             batch_size=10,
                                             eval_batch_size=10,
                                             num_workers=1)
    for loader in [train, test]:
        n_batches = 10
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            assert y.numpy().sum() == 0., "all targets should be class 0"


if __name__ == '__main__':
    _test_shannon_entropy()
    _test_filter_dataset()
