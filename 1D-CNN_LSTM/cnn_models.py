from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.python.keras.callbacks import TensorBoard
from model import Model
from data import Dataset
from metrics import F1_metrics
import os



class Simple_shallow_cnn(Model):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data, tb_log_dir=tb_log_dir)
        self.build()

    def evaluate(self, log_dir=None):
        accuracy = super().evaluate()

        return accuracy

    def build(self):
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', dilation_rate=1, input_shape=(self.n_timesteps, self.n_features)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', dilation_rate=1))
        model.add(Flatten())
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        super().build(model)


class Wavenet_deep_cnn(Model):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data,
                         tb_log_dir=tb_log_dir, batch_size=64)
        self.build()

    def evaluate(self, log_dir=None):
        accuracy = super().evaluate()

        return accuracy

    def build(self):
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=1, input_shape=(self.n_timesteps, self.n_features)))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', dilation_rate=2))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', dilation_rate=4))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', dilation_rate=8))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', dilation_rate=16))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', dilation_rate=32))
        model.add(Flatten())
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        super().build(model)


if __name__ == "__main__":
    path = os.getcwd()
    dataset_path = '/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3_term_project/project/LSTM-Human-Activity-Recognition-master/data/UCI HAR Dataset'
    # dataset_root = path + dataset_path
    dataset_root = dataset_path

    print(dataset_root)
    dataset = Dataset(dataset_root=dataset_root)
    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")
