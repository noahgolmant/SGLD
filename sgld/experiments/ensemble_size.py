""" This experiement analyzes the difference in the efffect of the ensemble sizes.

1) We explore the change in accuracy as a function of the size of the ensemble.
2) We explore the variance of the individual models accuracy as a function of the
size of the ensemble.
3) We explore how the entropy of a predicted class varies as a function of the size
of the ensemble.    """

import itertools
from skeletor.datasets import build_dataset
from torch.nn import CrossEntropyLoss
import track

from sgld.train import test  # the evaluate function
from sgld.utils import SoftmaxNLL, Entropy, build_single_class_dataset
from sgld.ensemble import Ensemble


def run(ensemble, trial_df, results_dir='./logs', dataroot='./data',
        class_ind=0, batch_size=128, eval_batch_size=100, cuda=False,
        num_workers=2, start_epoch=160, end_epoch=200, **unused):

    trainloader, testloader = build_dataset('cifar10',
                                            dataroot=dataroot,
                                            batch_size=batch_size,
                                            eval_batch_size=eval_batch_size,
                                            num_workers=2)

    # this will only iterate over examples of one class
    class_trainlaoder, class_testloader = build_single_class_dataset(
        'cifar10',
        class_ind=class_ind,
        dataroot=dataroot,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=2)

    full_ensemble = ensemble
    for i in range(len(ensemble.models) - 1, -1, -1):
        ensemble_size = len(ensemble.models) - i
        ensemble_loss = SoftmaxNLL()
        one_loss = CrossEntropyLoss()

        entropy_criterion = Entropy()

        ensemble = Ensemble(full_ensemble.models[i:])
        single_model = full_ensemble.models[i]

        # we want to do metrics for (a) the ensemble with varying sizes and
        #   (b) the individual models corresponding to that epoch
        def _test_dataset(model, testloader, criterion):
            loss, acc = test(testloader, model,
                             criterion, epoch=-1,
                             cuda=cuda, metric=False)
            # compute the entropy of the model post-hoc as well
            entropy = test(testloader, model,
                           entropy_criterion, epoch=-1,
                           cuda=cuda, metric=False,
                           criterion_has_labels=False,
                           compute_acc=False)
            return loss, acc, entropy
        # metrics for the both models over both datasets
        # (a) on the whole dataset
        #      (i) for the ensemble
        #      (ii)for the single model from this epoch
        # (b) on the whole dataset
        #      (i) for the ensemble
        #      (ii)for the single model from this epoch
        stats = {}
        models = (ensemble, single_model)
        loaders = (testloader, class_testloader)
        losses = ensemble_loss, one_loss
        model_names = ['ensemble', 'single_model']
        loader_names = ['full', 'single_class']
        for i, j in itertools.product(range(len(models)), range(len(loaders))):
            track.debug("[ensemble size: %d] Evaluating loss/acc/entropy for"
                        "%s on %s dataset" %
                        (ensemble_size, model_names[i], loader_names[i]))
            metric = model_names[i] + '_' + loader_names[i]
            loss, acc, entropy = _test_dataset(models[i], loaders[j], losses[i])
            stats[metric + '_loss'] = loss
            stats[metric + '_acc'] = acc
            stats[metric + '_entropy'] = entropy
        track.metric(ensemble_size=ensemble_size, **stats)
