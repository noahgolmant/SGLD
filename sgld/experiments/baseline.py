"""
this should do standard accuracy / postprocessing for a model,
e.g. just plotting test accuracy over time in the training process


goal plot: standard loss curve for the baseline as one line
dashed horizontal lines for: baseline max test acc, ensemble max test acc
"""
from skeletor.datasets import build_dataset
import torch
import track

from sgld.analysis import load_trial
from sgld.train import test  # the evaluate function
from sgld.utils import SoftmaxNLL


def run(ensemble, trial_df, results_dir='./logs', dataroot='./data',
        batch_size=128, eval_batch_size=100, cuda=False, num_workers=2,
        **unused):
    """
    this evaluates both the ensemble and the baseline model on the full
    test set

    we also evaluate each model and compute their individual losses, so that
    we can plot the variance around the ensemble's dashed horizontal line
        (see top of file)
    """
    trainloader, testloader = build_dataset('cifar10',
                                            dataroot=dataroot,
                                            batch_size=batch_size,
                                            eval_batch_size=eval_batch_size,
                                            num_workers=2)
    ensemble_criterion = SoftmaxNLL()
    ensemble_loss, ensemble_acc = test(testloader, ensemble,
                                       ensemble_criterion, epoch=-1,
                                       cuda=cuda, metric=False)

    # get the no-noise baseline evaluation
    proj = track.Project(results_dir)
    best_model, best_df = load_trial(proj, noise_scale=0.0)

    baseline_criterion = torch.nn.CrossEntropyLoss()
    baseline_loss, baseline_acc = test(testloader, best_model,
                                       baseline_criterion,
                                       epoch=-1, cuda=cuda, metric=False)

    # now, test each of the ensemble's models
    model_losses = []
    model_accs = []
    for model in ensemble.models:
        model_loss, model_acc = test(testloader, model,
                                     baseline_criterion,
                                     epoch=-1, cuda=cuda, metric=False)
        model_losses.append(model_loss)
        model_accs.append(model_acc)

    # we just need to track the scalar results of this evaluation
    # we can access the baseline test *curve* from the jupyter notebook (later)
    track.metric(iteration=0, ensemble_loss=ensemble_loss,
                 ensemble_acc=ensemble_acc,
                 best_baseline_loss=baseline_loss,
                 best_baseline_acc=baseline_acc,
                 model_losses=model_losses,
                 model_accs=model_accs)
