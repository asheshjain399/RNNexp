import numpy as np
from maneuver_prediction_rnn import evaluate
import sys
import neuralmodels.predictions
import pickle
import os
import pandas as pd
import click

pd.set_option('display.width', 1300)

@click.group()
def cli():
    pass

@click.command()
@click.option('--save_file', prompt='Location of save file')
def display(save_file):
    results = pickle.load(open(save_file))

    for th in np.sort(results.keys()):
        print 'Threshold: %f' %  th
        print '(precision, recall)'
        print results[th]
        print '******************************'

# Evaluates with various checkpoints and output thresholds to determine
# most optimal prediction parameters
@click.command()
@click.option('--save_file', prompt='Save file location')
@click.option('--maneuver_type', prompt='Maneuver type')
@click.option('--idx', prompt='Data index')
def run(save_file, maneuver_type, idx):
    # Configure these
    checkpoints_params = np.append(np.arange(300, 599, 50), 599)
    thresh_params = np.arange(.6, .9, .01)

    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    results = {}

    for th in thresh_params:
        results_th = {}
        for fold in folds:
            for cp in checkpoints_params:
                print 'Predicting: th: %f, fold: %s, checkpoint: %s' % (th, fold, cp)

                with open('settings.py','w') as f:
                    f.write('OUTPUT_THRESH = %f \n' % th)
                import neuralmodels.predictions

                [conMat, p_mat, re_mat, time_mat] = evaluate(
                    idx, fold, cp, 'multipleRNNs', maneuver_type)

                precision = np.mean(np.diag(p_mat)[1:])
                recall = np.mean(np.diag(re_mat)[1:])

                results_th[(fold, cp)] = [(precision, recall)]
        df_th = pd.DataFrame(results_th).stack().reset_index(level=0, drop = True)
        df_th['mean'] = df_th.apply(lambda x: (np.mean([x[i][0] for i in xrange(len(x))]), np.mean([x[i][1] for i in xrange(len(x))])), axis=1)
        print(df_th)

        results[th] = df_th

    pickle.dump(results, open(save_file, 'wb'))

cli.add_command(run)
cli.add_command(display)

if __name__ == '__main__':
    cli()
