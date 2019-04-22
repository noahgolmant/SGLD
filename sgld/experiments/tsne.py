"""
this experiment should visualize the feature maps for various examples,
with one blob for each model in the ensemble
"""
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from skeletor.utils import progress_bar
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
    labels = []

    track.debug("[tsne] starting forward passes")
    ensemble.models = ensemble.models[0::4]  # plot every 4 epochs for now
    for model_ind, model in enumerate(ensemble.models):
        # plot progress
        progress_bar(model_ind, len(ensemble.models))
        model_activations = []
        # this hook will aggregate a list of model outputs in `activations`
        model.linear.register_forward_pre_hook(
            _create_preactivation_hook(model_activations))

        with torch.no_grad():
            for inputs, _ in islice(trainloader, 0, num_batches):
                model(inputs)
        train_activations[model_ind] = torch.cat(model_activations)
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
    # plot the model means too
    model_means = []
    num_model_vecs = len(list(train_activations.values())[0])

    endpoints = []
    start = 0
    for stop in range(0, len(embedding), num_model_vecs):
        if stop - start > 0:
            endpoints.append((start, stop))
            start = stop
    for start, stop in endpoints:
        model_means.append(embedding[start:stop, :].mean(axis=0))
    model_means = np.array(model_means)
    ys = np.array(list(range(len(model_means)))) / float(len(model_means))
    plt.scatter(model_means[:, 0], model_means[:, 1], c=ys, s=100,
                linewidth=2, edgecolors='black', marker='D')

    plt.axis('off')
    plt.savefig('/Users/noah/Dev/SGLD/embeddings.png', bbox_inches='tight')
    plt.close(f)
    track.debug("[tsne] done! saved to embeddings.jpg")
