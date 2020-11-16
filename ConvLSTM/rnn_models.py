'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 9:38:33 pm
Author: ankurrc
'''
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D
from tensorflow.python.keras.callbacks import TensorBoard

from model import Model
from model_2 import Model2
from data import Dataset
from metrics import F1_metrics
from keras.layers.merge import Concatenate, concatenate
# from keras.models import Model

from keras.layers import Input


class LSTM_model(Model):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data, tb_log_dir=tb_log_dir)
        self.build()

    def evaluate(self, log_dir=None):
        accuracy = super().evaluate()
        # print("accuracy")
        # print(accuracy)
        return accuracy

    def build(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(
            self.n_timesteps, self.n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        super().build(model)


class CNN_LSTM_model(Model):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data,
                         tb_log_dir=tb_log_dir, batch_size=64)
        self.build()

    def evaluate(self, log_dir=None):
        accuracy = super().evaluate()

        return accuracy

    def build(self):
        n_steps, n_length = 4, 32

        self.train_X = self.train_X.reshape(
            (self.train_X.shape[0], n_steps, n_length, self.n_features)) # self.n_features = 6
        self.test_X = self.test_X.reshape(
            (self.test_X.shape[0], n_steps, n_length, self.n_features))

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3,
                                         activation='relu'), input_shape=(None, n_length, self.n_features)))
        model.add(TimeDistributed(
            Conv1D(filters=16, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        # print(model.summary())
        # input()

        super().build(model)


class CNN_LSTM_model_2(Model2):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data,
                         tb_log_dir=tb_log_dir, batch_size=64)
        self.build()

    def evaluate(self, log_dir=None, model=None):

        # model.add(Dense(self.n_outputs, activation='softmax'))
        # model.compile(loss='categorical_crossentropy',
        #                    optimizer='adam', metrics=['accuracy'])

        # # print(model.summary())
        # # input()

        # super().build(model)

        accuracy = super().evaluate()

        return accuracy

    def build(self):
        n_steps, n_length = 4, 32

        model = Sequential()
        model2 = Sequential()
        for i in range (2):

            if i == 0:

                self.train_X = self.train_X.reshape(
                    (self.train_X.shape[0], n_steps, n_length, self.n_features)) # self.n_features = 6
                self.test_X = self.test_X.reshape(
                    (self.test_X.shape[0], n_steps, n_length, self.n_features))

                # model = Sequential()
                model.add(TimeDistributed(Conv1D(filters=64, kernel_size=12,
                                                activation='relu'), input_shape=(None, n_length, self.n_features)))
                model.add(TimeDistributed(
                    Conv1D(filters=16, kernel_size=3, activation='relu')))
                model.add(TimeDistributed(Dropout(0.5)))
                model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
                model.add(TimeDistributed(Flatten()))
                model.add(LSTM(100))
                model.add(Dropout(0.5))
                model.add(Dense(100, activation='relu'))
                # model.add(Dense(self.n_outputs, activation='softmax'))
            
            else:

                self.train_X2 = self.train_X2.reshape(
                    (self.train_X2.shape[0], n_steps, n_length, self.n_features)) # self.n_features = 6
                self.test_X2 = self.test_X2.reshape(
                    (self.test_X2.shape[0], n_steps, n_length, self.n_features))

                # model2 = Sequential()
                model2.add(TimeDistributed(Conv1D(filters=64, kernel_size=12,
                                                activation='relu'), input_shape=(None, n_length, self.n_features)))
                model2.add(TimeDistributed(
                    Conv1D(filters=16, kernel_size=3, activation='relu')))
                model2.add(TimeDistributed(Dropout(0.5)))
                model2.add(TimeDistributed(MaxPooling1D(pool_size=2)))
                model2.add(TimeDistributed(Flatten()))
                model2.add(LSTM(100))
                model2.add(Dropout(0.5))
                model2.add(Dense(100, activation='relu'))
                # model2.add(Dense(self.n_outputs, activation='softmax'))


        # x = concatenate([model, model2])
        # x = Dense(self.n_outputs, activation='softmax')(x)

        # concatenate them
        merged = Concatenate([model, model2])

        print(merged)

        # self.train_X = self.train_X.reshape(
        #             (self.train_X.shape[0], n_steps, n_length, self.n_features))

        re_train_x = Input(shape=(None, n_length, self.n_features))
        re_train_X2 = Input(shape=(None, n_length, self.n_features))

        # self.train_X2 = self.train_X2.reshape(
        #             (self.train_X2.shape[0], n_steps, n_length, self.n_features)) # self.n_features = 6

        big_model = Model(inputs=[re_train_x, re_train_X2], outputs=merged)

        big_model.add(Dense(self.n_outputs, activation='softmax'))

        # ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)

        big_model.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=['accuracy'])



        # model.add(concatenate([model, model2]))
        # model.add(Dense(self.n_outputs, activation='softmax'))
        # model.compile(loss='categorical_crossentropy',
        #                    optimizer='adam', metrics=['accuracy'])

        super().build(big_model)


class ConvLSTM_model(Model):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data,
                         tb_log_dir=tb_log_dir, batch_size=64)
        self.build()

    def evaluate(self, log_dir=None):
        accuracy = super().evaluate()

        return accuracy

    def build(self):
        n_steps, n_length = 4, 32

        self.train_X = self.train_X.reshape(
            (self.train_X.shape[0], n_steps, 1, n_length, self.n_features))
        self.test_X = self.test_X.reshape(
            (self.test_X.shape[0], n_steps, 1, n_length, self.n_features))

        # define model
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3),
                             activation='relu', input_shape=(n_steps, 1, n_length, self.n_features)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        # print(model.summary())
        # input()
        
        super().build(model)


if __name__ == "__main__":
    dataset_root = "/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3_term_project/project/human_activity_recognition-master/code/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3_term_project/project/LSTM-Human-Activity-Recognition-master/data/UCI HAR Dataset"
    dataset = Dataset(dataset_root=dataset_root)
    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    # lstm = LSTM(train_data={"X": train_X, "y": train_y},
    #             test_data={"X": test_X, "y": test_y})
