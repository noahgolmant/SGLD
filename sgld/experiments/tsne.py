"""
this experiment should visualize the feature maps for various examples,
with one blob for each model in the ensemble
"""
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from skeletor.utils import progress_bar
from skeletor.datasets import build_dataset
from sklearn import manifold
from scipy.spatial.distance import squareform
import torch
import track

from sgld.topicsne.tsne import TSNE
from sgld.topicsne.wrapper import Wrapper
from sgld.utils import build_single_class_dataset


def _create_preactivation_hook(activations):
    """
    when we add this hook to a model's layer, it is called whenever
    it is about to make the forward pass
    """
    def _linear_preactivation_hook(module, inputs):
        activations.append(inputs[0].cpu())
    return _linear_preactivation_hook


def _pairwise_distances(vecs, squared=False):
    """
    computes matrix A where A_ij = distance(vec i, vec j)
    vecs = (N x D) => returns (N x N) matrix
    """
    norms = torch.einsum('ij,ij->i', vecs, vecs)[:, None]
    distances = norms + norms.t() - 2 * (vecs @ vecs.t())
    if not squared:
        distances = torch.sqrt(distances)
    return distances


def _compute_densities(vecs, perplexity=30):
    distances = _pairwise_distances(vecs).numpy()
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances, perplexity, False)
    # Convert to n x n prob array
    pij = squareform(pij)
    return pij


def tsne_embeddings(vecs, train_iters, batch_size, perplexity=30, cuda=False):
    track.debug("[track]\tComputing image densities PMF")
    densities = _compute_densities(vecs, perplexity=perplexity)
    i, j = np.indices(densities.shape)
    i = i.ravel()
    j = j.ravel()

    track.debug("[track]\tTraining the TSNE embedding")
    tsne = TSNE(len(densities), 2, 2)  # visualize in 2d
    tsne_train_wrapper = Wrapper(tsne, batchsize=batch_size, cuda=cuda)
    for k in range(train_iters):
        # plot progress
        progress_bar(k, train_iters)
        tsne_train_wrapper.fit(densities, i, j)
    return tsne.logits.weight.detach().cpu().numpy()


def run(ensemble, proj_df, dataroot='./data',
        batch_size=128, cuda=False,
        class_ind=0,
        num_batches=4, tsne_train_iters=4000,
        **kwargs):
    """ let's do some dimensionality reduction """
    track.debug("[tsne] starting experiment with class %d" % class_ind)
    trainloader, testloader = build_single_class_dataset(
        'cifar10',
        class_ind=2,
        dataroot=dataroot,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=2)

    # stores for any loader; we have to copy these to the last two dicts
    train_activations = {}
    test_activations = {}
    labels = []

    track.debug("[tsne] starting forward passes")
    ensemble.models = ensemble.models[0::4]
    for model_ind, model in enumerate(ensemble.models):
        # plot progress
        progress_bar(model_ind, len(ensemble.models))
        model_activations = []
        # this hook will aggregate a list of model outputs in `activations`
        model.linear.register_forward_pre_hook(
            _create_preactivation_hook(model_activations))

        def _store_preactivations(loader):
            # each call will append to `model_activations`
            with torch.no_grad():
                for inputs, _ in islice(loader, 0, num_batches):
                    model(inputs)
            return torch.cat(model_activations)
        train_activations[model_ind] = _store_preactivations(trainloader)
        # test_activations[model_ind] = _store_preactivations(testloader)
        labels.extend([model_ind] * len(train_activations[model_ind]))

    # now, we have all activations for all models! we can do tsne
    track.debug("[tsne] forward pass done! starting stacking + embedding")
    all_train_activations = torch.cat(
        [vec for vec in train_activations.values()])
    embedding = tsne_embeddings(all_train_activations,
                                tsne_train_iters,
                                batch_size=len(all_train_activations),
                                cuda=cuda)

    f = plt.figure()
    # create labels for the models by iteration
    y = np.array(labels)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y * 1.0 / y.max())
    plt.axis('off')
    plt.savefig('/Users/noah/Dev/SGLD/embeddings.png', bbox_inches='tight')
    plt.close(f)
    track.debug("[tsne] done! saved to embeddings.jpg")
