'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 9:23:41 pm
Author: ankurrc
'''
import numpy as np

from metrics import F1_metrics
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


class Model2(object):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None, batch_size=128, epochs=100, verbosity=1): # epochs=25
        self.verbose = verbosity
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_timesteps = train_data["X"].shape[1]
        # self.n_features = (train_data["X"].shape[2])
        self.n_features = 3
        self.n_outputs = train_data["y"].shape[1]

        self.train_X = (train_data["X"])[:,:,:3]
        self.train_y = train_data["y"]
        self.test_X = (test_data["X"])[:,:,:3]
        self.test_y = test_data["y"]

        self.train_X2 = (train_data["X"])[:,:,3:]
        self.train_y2 = train_data["y"]
        self.test_X2 = (test_data["X"])[:,:,3:]
        self.test_y2 = test_data["y"]

        self.callbacks = [F1_metrics(), TensorBoard(
            log_dir=tb_log_dir, write_grads=True, write_graph=True, histogram_freq=3, batch_size=self.batch_size)]

        self.model = None

    def build(self, model):
        self.model = model

    def evaluate(self):
        self.model.fit(self.train_X, self.train_y, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=self.verbose, validation_split=0.2, callbacks=self.callbacks, shuffle=True)

        # evaluate model
        pred_y = self.model.predict(
            self.test_X, batch_size=self.batch_size, verbose=self.verbose)

        pred_y = np.argmax(pred_y, axis=1)
        target_y = np.argmax(self.test_y, axis=1)

        acc_score = accuracy_score(target_y, pred_y)

        # print(pred_y, target_y)

        precision, recall, f1, _ = precision_recall_fscore_support(
            target_y, pred_y, average="weighted")
        return precision, recall, f1, acc_score
