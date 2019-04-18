""" This experiement analyzes the difference in the efffect of the ensemble sizes.

1) We explore the change in accuracy as a function of the size of the ensemble.
2) We explore the variance of the individual models accuracy as a function of the
size of the ensemble.
3) We explore how the entropy of a predicted class varies as a function of the size
of the ensemble.    """

import track
from skeletor.datasets import build_dataset
import torch

from sgld.analysis import load_trial
from sgld.train import test  # the evaluate function
from sgld.utils import SoftmaxNLL


def run(ensemble, trial_df, results_dir='./logs', dataroot='./data',
        batch_size=128, eval_batch_size=100, cuda=False, num_workers=2,
        start_epoch=160, end_epoch=200, **unused):

    trainloader, testloader = build_dataset('cifar10',
                                            dataroot=dataroot,
                                            batch_size=batch_size,
                                            eval_batch_size=eval_batch_size,
                                            num_workers=2)

    trial_start_epoch = end_epoch;
    model_accs = []

    for i in range(len(enseble.models) - 1, -1, -1):
        #Get ensemble model where size = end_epoch - trial_start_epoch
        ensemble_size = len(enseble.models) - i
        ensemble_trial = Ensemble(ensemble_models[i:])

        ensemble_criterion = SoftmaxNLL()
        ensemble_loss, ensemble_acc = test(testloader, ensemble_trial,
                                           ensemble_criterion, epoch=-1,
                                           cuda=cuda, metric=False)

        # Since we are iteratively increasing our ensemble size by one, we can
        # exlusively look at the model being added to add to our list of
        # model accuracies
        model = ensemble.models[i]
        model_loss, model_acc = test(testloader, model,
                                     baseline_criterion,
                                     epoch=-1, cuda=cuda, metric=False)
        model_accs.append(model_acc)

        ensemble_acc_var = np.var(np.array(model_acc))


       track.metric(ensemble_size=ensemble_size, ensemble_loss=ensemble_loss,
                    ensemble_acc=ensemble_acc, ensemble_acc_var=ensemble_acc_var )
