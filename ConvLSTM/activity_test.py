# 10-folds and All Includes
# Oject

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import os
import sys
import io
import copy

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
from keras.models import load_model
from sklearn.metrics import accuracy_score


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
    
    # y_ = np.reshape(np.array(y_), [-1, 1])
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


def LSTM_RNN(_X, _weights, _biases, n_input, n_hidden, n_steps):

    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset, some of the credits goes to 
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier, 
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_, n_classes):
    # Function to encode neural one-hot output labels from number indexes 
    # e.g.: 
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def training(DATA_PATH):

    # DATA_PATH = "data/"

    os.chdir(DATA_PATH)
    os.chdir("..")
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
        _y_test.append(np.array(r_y_test))
        _X_train.append(np.array(r_X_train))
        _y_train.append(np.array(r_y_train))

    ###########################################################################

    accuracy_list = []
    for i in range(10):

        tf.reset_default_graph()

        X_train = np.array(_X_train[i])
        X_test = np.array(_X_test[i])
        y_train = np.array(_y_train[i])
        y_test = np.array(_y_test[i])

        # Input Data 
        training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
        test_data_count = len(X_test)  # 2947 testing series
        n_steps = len(X_train[0])  # 128 timesteps per series
        n_input = len(X_train[0][0])  # 6 input parameters per timestep

        # LSTM Neural Network's internal structure
        n_hidden = 32 # Hidden layer num of features
        n_classes = 6 # Total classes (should go up, or should go down)

        # Training 
        learning_rate = 0.0025
        lambda_loss_amount = 0.0015
        training_iters = training_data_count * 300  # Loop 300 times on the dataset
        # training_iters = training_data_count * 1  # Loop 300 times on the dataset
        batch_size = 1500
        display_iter = 30000  # To show test set accuracy during training

        # Some debugging info
        print("Some useful info to get an insight on dataset's shape and normalisation:")
        print("(X shape, y shape, every X's mean, every X's standard deviation)")
        print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
        print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

        # Graph input/output
        x = tf.placeholder(tf.float32, [None, n_steps, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])

        # Graph weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([n_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        pred = LSTM_RNN(x, weights, biases, n_input, n_hidden, n_steps)

        # Loss, optimizer and evaluation
        l2 = lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        ) # L2 loss prevents this overkill neural network to overfit the data
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        ################################### trainging performance ###################################

        # To keep track of training's performance
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []

        saver = tf.train.Saver()

        # Launch the graph
        sess = tf.InteractiveSession()
        # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        # init = tf.local_variables_initializer()
        sess.run(init)

        # Perform Training steps with "batch_size" amount of example data at each loop
        step = 1
        while step * batch_size <= training_iters:
            batch_xs =         extract_batch_size(X_train, step, batch_size)
            batch_ys = one_hot(extract_batch_size(y_train, step, batch_size), n_classes)

            # Fit training using batch data
            _, loss, acc = sess.run(
                [optimizer, cost, accuracy],
                feed_dict={
                    x: batch_xs, 
                    y: batch_ys
                }
            )
            train_losses.append(loss)
            train_accuracies.append(acc)
            
            # Evaluate network only at some steps for faster training: 
            if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
                
                # To not spam console, show training accuracy/loss in this "if"
                print("Training iter #" + str(step*batch_size) + \
                    ":   Batch Loss = " + "{:.6f}".format(loss) + \
                    ", Accuracy = {}".format(acc))
                
                # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
                loss, acc = sess.run(
                    [cost, accuracy], 
                    feed_dict={
                        x: X_test,
                        y: one_hot(y_test, n_classes)
                    }
                )
                test_losses.append(loss)
                test_accuracies.append(acc)
                print("PERFORMANCE ON TEST SET: " + \
                    "Batch Loss = {}".format(loss) + \
                    ", Accuracy = {}".format(acc))

            step += 1
            # break

        saver.save(sess, "./my_test_model", global_step=1000)

        print("Optimization Finished!")

        # Accuracy for test data

        one_hot_predictions, accuracy, final_loss = sess.run(
            [pred, accuracy, cost],
            feed_dict={
                x: X_test,
                y: one_hot(y_test, n_classes)
            }
        )

        test_losses.append(final_loss)
        test_accuracies.append(accuracy)

        accuracy_list.append(accuracy)

        print("FINAL RESULT: " + \
            "Batch Loss = {}".format(final_loss) + \
            ", Accuracy = {}".format(accuracy))

        # Results
        predictions = one_hot_predictions.argmax(1)

        print("Testing Accuracy: {}%".format(100*accuracy))

        print("")
        print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
        print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
        print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

        print("")
        print("Confusion Matrix:")
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        print(confusion_matrix)
        normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

        print("")
        print("Confusion matrix (normalised to % of total test data):")
        print(normalised_confusion_matrix)
        print("Note: training and testing data is not equally distributed amongst classes, ")
        print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

        # draw plot
        # plot_1(batch_size, train_losses, train_accuracies, test_losses, test_accuracies, display_iter, training_iters)
        # plot_2(normalised_confusion_matrix, n_classes)

        sess.close()

        # if i == 0:
        #     break

    acc_len = len(accuracy_list)
    acc_sum = 0
    for i in accuracy_list:
        acc_sum = acc_sum + i
    print("** 10-folds cv result **")
    print(accuracy_list)
    print(acc_sum/acc_len)


def training_all(DATA_PATH):

    # DATA_PATH = "data/"

    os.chdir(DATA_PATH)
    os.chdir("..")
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

    X = load_X(X_signals_paths) # total X
    y = load_y(X_signals_paths) # total y
    print(X.shape)
    print(y.shape)
    # input()

    # # 10-folds cross validation
    # _X_train = []
    # _X_test = []
    # _y_train = []
    # _y_test = []
    # total_num = X.shape[0]
    # num = int(total_num/10)
    # print(num)
    # init_num = num

    # for i in range (10):
    #     num = init_num * i
    #     # print(num)
    #     # print(num+init_num)
    #     _X_test.append(X[num:num+init_num])
    #     _y_test.append(y[num:num+init_num])
    #     _X_train.append(np.concatenate((X[:num], X[init_num+num:]), axis=0))
    #     _y_train.append(np.concatenate((y[:num], y[init_num+num:]), axis=0))

    # accuracy_list = []
    # for i in range(10):

    tf.reset_default_graph()

    X_train = np.array(X)
    # X_test = np.array(_X_test[i])
    y_train = np.array(y)
    # y_test = np.array(_y_test[i])

    print("check dataset shape")
    print(X_train.shape)
    print(y_train.shape)

    # Input Data 
    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
    # test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 6 input parameters per timestep

    # LSTM Neural Network's internal structure
    n_hidden = 32 # Hidden layer num of features
    n_classes = 6 # Total classes (should go up, or should go down)

    # Training 
    learning_rate = 0.0025
    lambda_loss_amount = 0.0015
    training_iters = training_data_count * 300  # Loop 300 times on the dataset
    # training_iters = training_data_count * 1  # Loop 300 times on the dataset
    batch_size = 1500
    display_iter = 30000  # To show test set accuracy during training

    # Some debugging info
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    # print("(X shape, y shape, every X's mean, every X's standard deviation)")
    # print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases, n_input, n_hidden, n_steps)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    ) # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ################################### trainging performance ###################################

    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    saver = tf.train.Saver()

    # Launch the graph
    sess = tf.InteractiveSession()
    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    # init = tf.local_variables_initializer()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    while step * batch_size <= training_iters:
        batch_xs =         extract_batch_size(X_train, step, batch_size)
        batch_ys = one_hot(extract_batch_size(y_train, step, batch_size), n_classes)

        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs, 
                y: batch_ys
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)
        
        # Evaluate network only at some steps for faster training: 
        if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
            
            # To not spam console, show training accuracy/loss in this "if"
            print("Training iter #" + str(step*batch_size) + \
                ":   Batch Loss = " + "{:.6f}".format(loss) + \
                ", Accuracy = {}".format(acc))
            
            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            # loss, acc = sess.run(
            #     [cost, accuracy], 
            #     feed_dict={
            #         x: X_test,
            #         y: one_hot(y_test, n_classes)
            #     }
            # )
            # test_losses.append(loss)
            # test_accuracies.append(acc)
            # print("PERFORMANCE ON TEST SET: " + \
            #     "Batch Loss = {}".format(loss) + \
            #     ", Accuracy = {}".format(acc))

        step += 1
        # break

    saver.save(sess, "./my_test_model", global_step=1000)

    print("Optimization Finished!")

    # Accuracy for test data

    # one_hot_predictions, accuracy, final_loss = sess.run(
    #     [pred, accuracy, cost],
    #     feed_dict={
    #         x: X_test,
    #         y: one_hot(y_test, n_classes)
    #     }
    # )

    # test_losses.append(final_loss)
    # test_accuracies.append(accuracy)

    # accuracy_list.append(accuracy)

    # print("FINAL RESULT: " + \
    #     "Batch Loss = {}".format(final_loss) + \
    #     ", Accuracy = {}".format(accuracy))

    # # Results
    # predictions = one_hot_predictions.argmax(1)

    # print("Testing Accuracy: {}%".format(100*accuracy))

    # print("")
    # print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
    # print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
    # print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

    # print("")
    # print("Confusion Matrix:")
    # confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    # print(confusion_matrix)
    # normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

    # print("")
    # print("Confusion matrix (normalised to % of total test data):")
    # print(normalised_confusion_matrix)
    # print("Note: training and testing data is not equally distributed amongst classes, ")
    # print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

    # draw plot
    # plot_1(batch_size, train_losses, train_accuracies, test_losses, test_accuracies, display_iter, training_iters)
    # plot_2(normalised_confusion_matrix, n_classes)

    sess.close()

    # acc_len = len(accuracy_list)
    # acc_sum = 0
    # for i in accuracy_list:
    #     acc_sum = acc_sum + i
    # print("** 10-folds cv result **")
    # print(accuracy_list)
    # print(acc_sum/acc_len)


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


def testing(DATA_PATH):

    # dataset_root = args.dataset
    # dataset_root = '/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3_term_project/project/LSTM-Human-Activity-Recognition-master/data/UCI HAR Dataset'
    # num_repeats = args.repeats
    # # num_repeats = 1
    # models = args.models

    # print(models)

    log_dir = "logs"
    results_dir = "results"
    models_dir = "models"

    #######################################################################

    # DATA_PATH = "/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3_term_project/project/LSTM-Human-Activity-Recognition-master/data/"
    # os.chdir(DATA_PATH)
    # os.chdir("..")
    DATASET_PATH = DATA_PATH + "Activity_Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)

    TRAIN = "train/"
    TEST = "test/"

    # read all file list in a directory
    # train_path = DATASET_PATH + TRAIN + "Inertial Signals/"
    test_path = DATASET_PATH + TEST + "Inertial Signals/"
    for r, d, f in os.walk(test_path):
        for file in f:
            if '.txt' in file:
                INPUT_SIGNAL_DATA.append(os.path.join(r, file))


    X_signals_paths = [
        signal for signal in INPUT_SIGNAL_DATA
    ]

    X = load_X(X_signals_paths) # total X
    y = load_y_dummy(X_signals_paths) # total y
    # y_true, _ = np.array(load_y(X_signals_paths)) # remove
    print(X.shape)
    # print(y.shape)
    # input()

    # # 10-folds cross validation
    # _X_train = []
    # _X_test = []
    # _y_train = []
    # _y_test = []
    # total_num = X.shape[0]
    # num = int(total_num/10)
    # print(num)
    # init_num = num

    # for i in range (10):
    #     num = init_num * i
    #     # print(num)
    #     # print(num+init_num)
    #     _X_test.append(X[num:num+init_num])
    #     _y_test.append(y[num:num+init_num])
    #     _X_train.append(np.concatenate((X[:num], X[init_num+num:]), axis=0))
    #     _y_train.append(np.concatenate((y[:num], y[init_num+num:]), axis=0))

    # accuracy_list = []
    # for i in range(10):

    # tf.reset_default_graph()

    X_test = np.array(X)
    y_test = np.array(y)
    # X_test = np.array(_X_test[i])
    # y_train = np.array(y)
    # y_test = np.array(_y_test[i])

    print("check dataset shape")
    print(X_test.shape)
    # print(y_train.shape)

    ensemble = []
    for i in range (10):
        temp = []
        ensemble.append(temp)
    # ensemble = np.array(ensemble)
    # print(ensemble)
    for i in range (10):
        model = load_model('../ConvLSTM/models/conv_lstm_{}.h5'.format(i))
       
        n_steps, n_length, n_features = 4, 32, 6

        # self.train_X = self.train_X.reshape(
        #     (self.train_X.shape[0], n_steps, n_length, n_features))
        X_test = X_test.reshape(
            (X_test.shape[0], n_steps, n_length, n_features))

        # print(model.predict(X_test).shape)
        probability = model.predict(X_test) # (n, 6)
        # probability = np.array(probability)
        ensemble[i].append(probability)
        # if i > 0:
        #     ensemble[i] = np.concatenate((ensemble[i], probability), 0)
        # else:
        #     ensemble[i] = probability

        # print(ensemble)

        pred = model.predict_classes(X_test)
        # print(np.array(pred).shape)
        # print(pred)
        # print(y_true)

        # acc = accuracy_score(y_true, pred)
        # print(acc)
    
    
    # for i in ensemble:
    ensemble = np.array(ensemble)
    ensemble = np.squeeze(ensemble)
    print(ensemble.shape)

    summed = np.sum(ensemble, axis=0)
    results = np.argmax(summed, axis=1)

    # print("final acc")
    # acc = accuracy_score(y_true, pred)
    # print(acc)

    # predictions = one_hot_predictions.argmax(1)
    print("FINAL RESULT: " + \
        "pred = {}".format(results))

    # write result in txt file
    with open('./result_annkyeongjin.txt', 'w') as result_file:
        for result in results:
            print(result + 1)
            result_file.write(str(result + 1) + '\n')


def testing_2(DATA_PATH):

    # Input Data 
    testing_data_count = len(X_test)  # 7352 training series (with 50% overlap between each serie)
    
    # test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_test[0])  # 128 timesteps per series
    n_input = len(X_test[0][0])  # 6 input parameters per timestep

    # LSTM Neural Network's internal structure
    n_hidden = 32 # Hidden layer num of features
    n_classes = 6 # Total classes (should go up, or should go down)

    # Training 
    learning_rate = 0.0025
    lambda_loss_amount = 0.0015
    # training_iters = training_data_count * 300  # Loop 300 times on the dataset
    testing_iters = testing_data_count * 1  # Loop 300 times on the dataset
    batch_size = 1500
    display_iter = 30000  # To show test set accuracy during training

    # Some debugging info
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    # print("(X shape, y shape, every X's mean, every X's standard deviation)")
    # print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases, n_input, n_hidden, n_steps)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    ) # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ################################### trainging performance ###################################

    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    saver = tf.train.Saver()

    # Launch the graph
    sess = tf.InteractiveSession()
    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    # init = tf.global_variables_initializer()
    # init = tf.local_variables_initializer()
    # sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    # step = 1
    # while step * batch_size <= training_iters:
    #     batch_xs =         extract_batch_size(X_train, step, batch_size)
    #     batch_ys = one_hot(extract_batch_size(y_train, step, batch_size), n_classes)

    #     # Fit training using batch data
    #     _, loss, acc = sess.run(
    #         [optimizer, cost, accuracy],
    #         feed_dict={
    #             x: batch_xs, 
    #             y: batch_ys
    #         }
    #     )
    #     train_losses.append(loss)
    #     train_accuracies.append(acc)
        
    #     # Evaluate network only at some steps for faster training: 
    #     if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
            
    #         # To not spam console, show training accuracy/loss in this "if"
    #         print("Training iter #" + str(step*batch_size) + \
    #             ":   Batch Loss = " + "{:.6f}".format(loss) + \
    #             ", Accuracy = {}".format(acc))
            
    #         # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
    #         # loss, acc = sess.run(
    #         #     [cost, accuracy], 
    #         #     feed_dict={
    #         #         x: X_test,
    #         #         y: one_hot(y_test, n_classes)
    #         #     }
    #         # )
    #         # test_losses.append(loss)
    #         # test_accuracies.append(acc)
    #         # print("PERFORMANCE ON TEST SET: " + \
    #         #     "Batch Loss = {}".format(loss) + \
    #         #     ", Accuracy = {}".format(acc))

    #     step += 1
    #     # break

    # saver.save(sess, "./my_test_model", global_step=1000)
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accuracy for test data

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: X_test,
            y: one_hot(y_true, n_classes)
        }
    )

    # test_losses.append(final_loss)
    # test_accuracies.append(accuracy)

    # accuracy_list.append(accuracy)

    predictions = one_hot_predictions.argmax(1)
    print("FINAL RESULT: " + \
        "pred = {}".format(predictions))

    # write result in txt file
    with open('./result_annkyeongjin.txt', 'w') as result_file:
        for result in predictions:
            print(result + 1)
            result_file.write(str(result + 1) + '\n')

    # Results
    # predictions = one_hot_predictions.argmax(1)

    print("Testing Accuracy: {}%".format(100*accuracy))

    # print("")
    # print("Precision: {}%".format(100*metrics.precision_score(y_true, predictions, average="weighted")))
    # print("Recall: {}%".format(100*metrics.recall_score(y_true, predictions, average="weighted")))
    # print("f1_score: {}%".format(100*metrics.f1_score(y_true, predictions, average="weighted")))

    # print("")
    # print("Confusion Matrix:")
    # confusion_matrix = metrics.confusion_matrix(y_true, predictions)
    # print(confusion_matrix)
    # normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

    # print("")
    # print("Confusion matrix (normalised to % of total test data):")
    # print(normalised_confusion_matrix)
    # print("Note: training and testing data is not equally distributed amongst classes, ")
    # print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

    # draw plot
    # plot_1(batch_size, train_losses, train_accuracies, test_losses, test_accuracies, display_iter, training_iters)
    # plot_2(normalised_confusion_matrix, n_classes)

    sess.close()


############################################### plot ###############################################


def plot_1(batch_size, train_losses, train_accuracies, test_losses, test_accuracies, display_iter, training_iters):
    font = {
        'family' : 'Bitstream Vera Sans',
        'weight' : 'bold',
        'size'   : 18
    }
    matplotlib.rc('font', **font)

    width = 12
    height = 12
    plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
        [training_iters]
    )
    plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')

    plt.show()


def plot_2(normalised_confusion_matrix, n_classes):
    # Plot Results: 
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix, 
        interpolation='nearest', 
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main(argv):

    # training_mode = 1 # train
    training_mode = 0 # test

    DATA_PATH = "../DataSet/"

    if training_mode == 1:
        training(DATA_PATH)
        # training_all(DATA_PATH)
    else:
        testing(DATA_PATH)


if __name__ == "__main__":
   main(sys.argv[1:])

