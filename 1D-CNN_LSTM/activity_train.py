'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 10:55:41 pm
Author: ankurrc
'''
import numpy as np
import os
import argparse

from rnn_models import LSTM_model, CNN_LSTM_model, ConvLSTM_model
from cnn_models import Simple_shallow_cnn, Wavenet_deep_cnn
from data import Dataset

# from keras import backend as K
from tensorflow.python.keras import backend as K
import copy
from tensorflow.python.keras.utils import to_categorical


def summarize_results(precison, recall, f1, acc):
    f1_m, f1_s = np.mean(f1), np.std(f1)
    pre_m, pre_s = np.mean(precison), np.std(precison)
    re_m, re_s = np.mean(recall), np.std(recall)
    acc_m, acc_s = np.mean(acc), np.std(acc)

    print(
        'Precision: {:.5f} (+/-{:.5f}) \t Recall: {:.5f} (+/-{:.5f}) \t F1: {:.5f} (+/-{:.5f}) \t ACC: {:.5f} (+/-{:.5f})'.format(pre_m, pre_s, re_m, re_s, f1_m, f1_s, acc_m, acc_s))

    return (pre_m, pre_s), (re_m, re_s), (f1_m, f1_s), (acc_m, acc_s)


def run_experiment(repeats=1, model_type=None, train_data=None, test_data=None, tb_log_dir=None):

    f1s = []
    precisions = []
    recalls = []
    accs = []

    model = None
    for r in range(repeats):
        K.clear_session()
        _log_dir = os.path.join(tb_log_dir, "run_{}".format(r))
        model = get_model(name=model_type, log_dir=_log_dir,
                        train_data=train_data, test_data=test_data)


        precision, recall, f1, acc = model.evaluate(log_dir=_log_dir)
        print('>>>>> repeat #{}--> Precision: {:.5f}, Recall: {:.5f}, F1: {:.5f}, ACC: {:.5f}'.format(r +
                                                                                1, precision, recall, f1, acc))
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        accs.append(acc)
    # summarize results
    p, r, f1, a = summarize_results(precisions, recalls, f1s, accs)
    return p, r, f1, a, model


def get_model(name, log_dir=None, train_data=None, test_data=None):

    if name == 'lstm':
        model = LSTM_model(train_data=train_data,
                           test_data=test_data, tb_log_dir=log_dir)
    elif name == 'cnn_lstm':
        model = CNN_LSTM_model(train_data=train_data,
                               test_data=test_data, tb_log_dir=log_dir)
    elif name == 'conv_lstm':
        model = ConvLSTM_model(train_data=train_data,
                               test_data=test_data, tb_log_dir=log_dir)
    elif name == 'simple_cnn':
        model = Simple_shallow_cnn(train_data=train_data,
                                   test_data=test_data, tb_log_dir=log_dir)
    elif name == 'wavenet_cnn':
        model = Wavenet_deep_cnn(train_data=train_data,
                                 test_data=test_data, tb_log_dir=log_dir)
    else:
        raise KeyError("Model '{}' not implemented!".format(name))

    return model


# Useful Constants
INPUT_SIGNAL_DATA = []

# Output classes to learn how to classify
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
] 

# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.strip().split('\t') for row in file
            ]]
        )
        file.close()
    
    return np.array(X_signals)


# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):

    y_ = []
    pid_ = []

    for yy in y_path:
        y_.append(int(yy[-5])-1)
        pid_.append(int(yy[-8:-6]))
    
    y_ = np.reshape(np.array(y_), [-1, 1])
    # pid_ = np.reshape(np.array(pid_), [-1, 1])
    pid_ = np.array(pid_)

    # print(pid_)
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_, pid_


# Load "y" (the neural network's testing outputs)
def load_y_dummy(y_path):

    y_ = []

    for yy in y_path:
        y_.append(0) # dummy
    
    y_ = np.reshape(np.array(y_), [-1, 1])
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_


def main(args):
    # dataset_root = args.dataset
    # dataset_root = '/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3_term_project/project/LSTM-Human-Activity-Recognition-master/data/UCI HAR Dataset'
    num_repeats = args.repeats
    # num_repeats = 1
    models = args.models

    models = ['cnn_lstm']

    print(models)

    log_dir = "logs"
    results_dir = "results"
    models_dir = "models"

    # dataset = Dataset(dataset_root=dataset_root)
    # train_X, train_y = dataset.load()
    # test_X, test_y = dataset.load(split="test")

    # print(train_X.shape)
    # print(train_y.shape)

    ###########################################################################
    DATA_PATH = "../DataSet/"
    os.chdir(DATA_PATH)
    # os.chdir("..")
    DATASET_PATH = DATA_PATH + "Activity_Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)

    TRAIN = "train/"
    TEST = "test/"

    # read all file list in a directory
    train_path = DATASET_PATH + TRAIN + "Inertial Signals/"
    for r, d, f in os.walk(train_path):
        for file in f:
            if '.txt' in file:
                INPUT_SIGNAL_DATA.append(os.path.join(r, file))


    X_signals_paths = [
        signal for signal in INPUT_SIGNAL_DATA
    ]

    ################################# patient #################################

    X = load_X(X_signals_paths) # total X
    y, pid = load_y(X_signals_paths) # total y
    print(X.shape)
    print(y.shape)
    # input()

    data_list = []
    for i in range (20):
        temp = []
        data_list.append(temp)

    for i, _pid in enumerate (pid):
        index = _pid-1
        data_list[index].append(X[i])

    y_list = []
    for i in range (20):
        temp = []
        y_list.append(temp)

    for i, _pid in enumerate (pid):
        index = _pid-1
        y_list[index].append(y[i])

    for i in data_list:
        print(np.array(i).shape)

    for i in y_list:
        print(np.array(i).shape)

    # 10-folds cross validation
    _X_train = []
    _X_test = []
    _y_train = []
    _y_test = []
    total_num = len(data_list)
    print(total_num)
    num = int(total_num/10)
    print(num)
    init_num = num

    idx_list = []
    fix_list = []
    for i in range (total_num): # 20 patients
        idx_list.append(i)
    fix_list = copy.copy(idx_list)

    for i in range (10):

        idx_list = copy.copy(fix_list)

        print(idx_list)

        num = init_num * i

        r_test_idx = []
        for j in range (num,num+init_num):
            r_test_idx.append(j)
            idx_list.remove(j)

        r_train_idx = idx_list

        print(r_test_idx)
        print(r_train_idx)

        r_X_test = data_list[r_test_idx[0]]
        r_y_test = y_list[r_test_idx[0]]
        for j in r_test_idx[1:]:
            r_X_test = np.concatenate((r_X_test, data_list[j]), axis=0)
            r_y_test = np.concatenate((r_y_test, y_list[j]), axis=0)
        print(np.array(r_X_test).shape)
        print(np.array(r_y_test).shape)

        r_X_train = data_list[r_train_idx[0]]
        r_y_train = y_list[r_train_idx[0]]
        for j in r_train_idx[1:]:
            r_X_train = np.concatenate((r_X_train, data_list[j]), axis=0)
            r_y_train = np.concatenate((r_y_train, y_list[j]), axis=0)
        print(np.array(r_X_train).shape)
        print(np.array(r_y_train).shape)

        print("check train, test shape")
        print(np.array(r_X_train).shape)
        print(np.array(r_X_test).shape)

        _X_test.append(np.array(r_X_test))
        r_y_test = to_categorical(r_y_test) # categorical y
        _y_test.append(np.array(r_y_test))
        _X_train.append(np.array(r_X_train))
        r_y_train = to_categorical(r_y_train) # categorical y
        _y_train.append(np.array(r_y_train))

    ###########################################################################

    results_dir = '../1D-CNN_LSTM/results'
    models_dir = '../1D-CNN_LSTM/models'

    num_repeats = 1

    accuracy_list = []
    for i in range(10):
        
        K.clear_session()
        # tf.reset_default_graph()

        train_X = np.array(_X_train[i])
        test_X = np.array(_X_test[i])
        train_y = np.array(_y_train[i])
        test_y = np.array(_y_test[i])

        print("Train, Test data shape")
        print(train_X.shape)
        print(test_X.shape)
        # input()

        for model_type in models:
            print(">>>>>>>>>>>>> Running experiments for '{}'".format(model_type))
            _log_dir = os.path.join(log_dir, model_type)

            precision, recall, f1, acc, model = run_experiment(repeats=num_repeats, model_type=model_type, train_data={"X": train_X, "y": train_y},
                                                        test_data={"X": test_X, "y": test_y}, tb_log_dir=_log_dir)

            print(">>>>>>>>>>>>> Writing results for '{}'".format(model_type))
            save_model_name = model_type + "_" + str(i)
            # print(save_model_name)
            with open(os.path.join(results_dir, save_model_name + ".txt"), "w") as res:
                line = "{}:\n".format(model_type)
                line += "Precision: {:.5f} (+/-{:.5f}) \n Recall: {:.5f} (+/-{:.5f}) \n F1: {:.5f} (+/-{:.5f}) \n ACC: {:.5f} (+/-{:.5f})\n".format(precision[0], precision[1],
                                                                                                                        recall[0], recall[1], f1[0], f1[1], acc[0], acc[1])
                line += "--------------------------------------------------------------------------------------------------------------------------------- \n"
                res.writelines(line)
            accuracy_list.append(acc[0])
            print(">>>>>>>>>>>>> Saving the model: {}_{}.h5".format(model_type, i))

        # print(os.path.join(models_dir, model_type + ".h5"))
        save_model_name = model_type + "_" + str(i)
        model.model.save(os.path.join(models_dir, save_model_name + ".h5"))

        print("accuracy_list")
        print(accuracy_list)

    acc_sum = 0
    for _acc in accuracy_list:
        acc_sum = acc_sum + _acc
    print(acc_sum/10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run models on the UCI HAR dataset.")
    parser.add_argument(
        "--dataset", help="Root path to UCI HAR dataset", type=str, default="UCI HAR Dataset/")
    parser.add_argument(
        "--repeats", help="No. of repeats for each model", type=int, default=10)
    parser.add_argument("--models", help="List of models to evaluate on. Valid models are: [lstm, cnn_lstm, conv_lstm, simple_cnn, wavenet_cnn]",
                        nargs='+')

    args = parser.parse_args()
    main(args)
