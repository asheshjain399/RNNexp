import sys
import numpy as np
import theano
import os
from theano import tensor as T
from neuralmodels.utils import permute, load
from neuralmodels.costs import softmax_decay_loss, softmax_loss
from neuralmodels.models import RNN, MultipleRNNsCombined
from neuralmodels.predictions import (
    OutputMaxProb, OutputSampleFromDiscrete, OutputActionThresh)
from neuralmodels.layers import (
    softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures)
from neuralmodels.updates import Adagrad, RMSprop
import cPickle
from utils import confusionMat
from predictions import predictManeuver, predictLastTimeManeuver

if __name__ == '__main__':

    index = sys.argv[1]
    fold = sys.argv[2]
    maneuver_type = sys.argv[3]

    pwd = os.getcwd()
    path_to_dataset = '%s/checkpoints/%s/%s' % (pwd, maneuver_type, fold)
    path_to_checkpoints = '%s/checkpoints/%s/%s' % (pwd, maneuver_type, fold)

    test_data = cPickle.load(
        open('{1}/test_data_{0}.pik'.format(index, path_to_dataset)))
    Y_te = test_data['labels']
    X_te = test_data['features']

    actions = []
    if 'actions' in test_data:
        actions = test_data['actions']
    else:
        actions = ['end_action', 'lchange', 'rchange', 'lturn', 'rturn']
    # print X_te.shape
    # print Y_te.shape

    train_data = cPickle.load(
        open('{1}/train_data_{0}.pik'.format(index, path_to_dataset)))
    Y_tr = train_data['labels']
    X_tr = train_data['features']
    print X_tr.shape
    print Y_tr.shape

    print type(X_tr[0, 0, 0])

    num_train = X_tr.shape[1]
    num_test = len(X_te)
    len_samples = X_tr.shape[0]

    num_classes = int(np.max(Y_tr) - np.min(Y_tr) + 1)
    inputD = X_tr.shape[2]
    outputD = num_classes

    print 'Number of classes ', outputD
    print 'Feature dimension ', inputD

    epochs = 600
    batch_size = num_train
    learning_rate_decay = 0.97
    decay_after = 5
    step_size = 6e-4

    use_pretrained = False
    train_more = False
    global rnn

    architectures = ['lstm_one_layer', 'lstm_two_layers', 'multipleRNNs']
    model_type = 2

    if not use_pretrained:
        if not os.path.exists(path_to_checkpoints):
            os.mkdir(path_to_checkpoints)

        if not os.path.exists('{1}/{0}/'.format(index, path_to_checkpoints)):
            os.mkdir('{1}/{0}/'.format(index, path_to_checkpoints))

        trY = T.lmatrix()

        # Creating network layers
        if architectures[model_type] == 'lstm_one_layer':
            layers = [TemporalInputFeatures(inputD),
                      LSTM('tanh', 'sigmoid', 'orthogonal', 4, 32, None),
                      softmax(num_classes)]
            rnn = RNN(layers, softmax_decay_loss, trY, step_size, Adagrad())
            rnn.fitModel(X_tr, Y_tr, 1,
                         '{1}/{0}/'.format(index, path_to_checkpoints),
                         epochs, batch_size, learning_rate_decay, decay_after)

        elif architectures[model_type] == 'lstm_two_layers':
            layers = [TemporalInputFeatures(inputD),
                      LSTM('tanh', 'sigmoid', 'orthogonal', 4, 32, None),
                      LSTM('tanh', 'sigmoid', 'orthogonal', 4, 32, None),
                      softmax(num_classes)]
            rnn = RNN(layers, softmax_decay_loss, trY, step_size, Adagrad())
            rnn.fitModel(X_tr, Y_tr, 1,
                         '{1}/{0}/'.format(index, path_to_checkpoints),
                         epochs, batch_size, learning_rate_decay, decay_after)

        elif architectures[model_type] == 'multipleRNNs':
            road_features_dimension = 4

            layers_1 = [TemporalInputFeatures(road_features_dimension)]
            layers_2 = [TemporalInputFeatures(inputD-road_features_dimension),
                        LSTM('tanh', 'sigmoid', 'orthogonal', 4, 64, None)]
            output_layer = [simpleRNN(
                            'tanh',
                            'normal',
                            4, 64,
                            temporal_connection=False),
                            softmax(num_classes)]

            rnn = MultipleRNNsCombined([layers_1, layers_2],
                                       output_layer,
                                       softmax_decay_loss,
                                       trY,
                                       step_size,
                                       Adagrad())

            rnn.fitModel([X_tr[:, :, (inputD-road_features_dimension):],
                          X_tr[:, :, :inputD-road_features_dimension]],
                         Y_tr, 1,
                         '{1}/{0}/'.format(index, path_to_checkpoints),
                         epochs,
                         batch_size,
                         learning_rate_decay,
                         decay_after)
