"""
this experiment will test the sensitivity / confidence of the model
on out-of-distribution data (e.g. from another dataset)"
"""
import track

from sgld.train import test
from sgld.utils import Entropy, build_single_class_dataset


def run(ensemble, proj_df, dataroot='./data', batch_size=128,
        eval_batch_size=100, cuda=False, num_workers=2, **unused):
    """ let's compute that entropy baby """
    num_classes = 10  # build_dataset('cifar10') <- not worth computing rn
    entropy_criterion = Entropy()

    # iterate for all possible classes in dataset
    ensemble.models = ensemble.models[::10]
    for class_ind in range(num_classes):
        # build dataset per class
        track.debug("Evaluating entropy for class id: %d" %
                    (class_ind))
        class_trainlaoder, class_testloader = build_single_class_dataset(
            'svhn',
            class_ind=class_ind,
            dataroot=dataroot,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=2)

        # compute the entropy of the model post-hoc as well
        entropy = test(class_testloader, ensemble,
                       entropy_criterion, epoch=-1,
                       cuda=cuda, metric=False,
                       criterion_has_labels=False,
                       compute_acc=False)
        track.debug("\n\n\tEntropy: %.2f" % entropy)
        track.metric(svnh_class_id=class_ind, entropy=entropy)
