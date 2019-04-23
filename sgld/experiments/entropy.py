"""
this experiment should look at the entropy of the ensemble over different
classes in same dataset
"""
import track
from skeletor.datasets import build_dataset, num_classes



def run(ensemble, proj_df, dataroot='./data', batch_size=128, eval_batch_size=100,
    cuda=False, num_workers=2, **unused):
    """ let's compute that entropy baby """

    num_classes = build_dataset('cifar10')
    entropy_criterion = Entropy()

    # iterate for all possible classes in dataset
    for class_id in range(num_classes):
        # build dataset per class
        class_trainlaoder, class_testloader = build_single_class_dataset(
            'cifar10',
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
        track.debug("Evaluating entropy for class id: %d" %
                    (class_id))

        track.metric(cifar_class_id=class_id, entropy=entropy)
