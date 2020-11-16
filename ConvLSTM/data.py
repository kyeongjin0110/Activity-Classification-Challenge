'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 9:00:32 pm
Author: ankurrc
'''
import os

import pandas as pd
import numpy as np

from tensorflow.python.keras.utils import to_categorical


class Dataset(object):

    def __init__(self, dataset_root=None):
        # print("pass dataset")
        # print(dataset_root)
        # self.dataset_root = os.path.abspath(dataset_root)
        self.dataset_root = dataset_root

    def _load_data(self, file_path=None):
        """
        Load data into a (rows, columns) format numpy array.
        """
        data = pd.read_csv(file_path, delim_whitespace=True, header=None)
        return data.values

    def _load_group(self, root, filenames, prefix="train"):
        """
        Load a group of files and concatenate them. 
        Returns a (num_samples, time_steps, features) format numpy array.
        """
        data = []
        for filename in filenames:
            file_path = os.path.join(root, filename)
            data.append(self._load_data(file_path))

        # stack along axis-3; Equivalent to np.concatenate(a[:,:,np.newaxis], b[:, ;, np.newaxis], axis=2)
        data = np.dstack(data)
        return data

    def load(self, split="train"):
        """
        Loads X and y.
        """
        if split.lower() not in ["test", "train"]:
            raise AssertionError(
                "split should be either of 'train' or 'test'")

        files_root = os.path.join(
            self.dataset_root, "{prefix}/Inertial Signals/".format(prefix=split))

        filenames = os.listdir(files_root)
        # load X
        X = self._load_group(files_root, filenames)
        # load y
        label_file_path = os.path.join(
            self.dataset_root, "{prefix}/y_{prefix}.txt".format(prefix=split))
        y = self._load_data(label_file_path)
        y -= 1  # zero-base the labels
        y = to_categorical(y)

        return X, y


class HAPTDataset(object):
    def __init__(self):
        pass


if __name__ == "__main__":
    dataset_root = "/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3_term_project/project/LSTM-Human-Activity-Recognition-master/data/UCI HAR Dataset"
    print("main")
    print(dataset_root)
    dataset = Dataset(dataset_root=dataset_root)

    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
