'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 10:44:19 pm
Author: ankurrc
'''
import numpy as np
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support


class F1_metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_targ, val_predict, average="weighted")

        self.val_f1s.append(f1)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        print("— val_f1: {:f} — val_precision: {:f} — val_recall {:f}".format(
            f1, precision, recall))
        return

    
