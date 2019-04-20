"""
Main file to orchestrate experiment analysis functionsso that we can plot it
later (probably in jupyter)
"""
import importlib
import skeletor
import torch
import track

from .ensemble import Ensemble


EXPERIMENTS = ['all', 'ensemble_size', 'entropy', 'ood', 'tsne', 'baseline']


def make_args(parser):
    """ add the command-line arguments u need here """
    parser.add_argument('--mode', type=str,
                        choices=EXPERIMENTS,
                        help='which experiment analysis code to run!',
                        default='all')
    parser.add_argument('--start_epoch', default=161,
                        help='burn-in period length')
    parser.add_argument('--end_epoch', default=200,
                        help='stop collecting samples after this epoch')
    parser.add_argument('--noise_scale', default=1e-8,
                        help='choose samples from trial trained with this'
                             ' noise level')
    parser.add_argument('--results_dir', default='logs/',
                        help='location of the track results for all trials')
    parser.add_argument('--cuda', action='store_true',
                        help='if true, use ur gpu')

    """ ADD YOUR EXPERIMENT-SPECIFIC ARGUMENTS HERE """
    # these will all be passed in as kwargs into your run function
    # (see experiments/baseline.py for example)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='training batch size')
    parser.add_argument('--eval_batch_size', default=100, type=int,
                        help='test set  batch size (so it divided dataset len')


def load_trial(proj, start_epoch=160, end_epoch=200, noise_scale=0.0):
    """
    returns the ensemble model and the dataframe of trial results
    """
    # find the right trial ID for this noise scale
    proj_ids = proj.ids[proj.ids['noise_scale'] == noise_scale]['trial_id']
    assert len(proj_ids) == 1, "we only want one sgld experiment for now"
    trial_id = str(proj_ids.values[0])
    proj_df = proj.results([trial_id])

    # just return a single model for the baseline case
    if noise_scale == 0.0:
        model = torch.load(proj.fetch_artifact(trial_id, 'best.ckpt'),
                           map_location='cpu')
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        return model, proj_df

    # load all the models after burn-in period
    models = []
    for epoch in range(start_epoch, end_epoch):
        model = torch.load(proj.fetch_artifact(trial_id,
                                               'model%d.ckpt' % epoch),
                           map_location='cpu')
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        models.append(model)
    ensemble = Ensemble(models)
    return ensemble, proj_df


def main(args):
    """
    loads the model and trial data and runs the specified experiment(s)!
    """
    # load the project from track
    proj = track.Project(args.results_dir)

    # create the ensemble
    model, trial_df = load_trial(proj, args.start_epoch, args.end_epoch,
                                 args.noise_scale)

    # run the experiment
    def _run(experiment):
        track.debug('Starting to run experiment: %s' % experiment)
        experiment_module = 'sgld.experiments.' + experiment
        runner = getattr(importlib.import_module(experiment_module), 'run')
        runner(model, trial_df, **vars(args))

    if args.mode == 'all':
        for experiment in EXPERIMENTS:
            _run(experiment)
    else:
        _run(args.mode)


if __name__ == '__main__':
    skeletor.supply_args(make_args)
    skeletor.execute(main)
