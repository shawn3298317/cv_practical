import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout,Activation

from util import split_k_folds

import pdb

class Model():
    def __init__(self, filter_sizes, kernel_nums, strides, batch_size=2, training_mode=False):
        self.filter_sizes = filter_sizes
        self.kernel_nums = kernel_nums
        self.batch_size = batch_size
        self.strides = strides
        self.training_mode = training_mode
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        # K layer cnn; relu + pooling  TODO: add BatchNorm
        for i, tup in enumerate(zip(self.filter_sizes, self.kernel_nums, self.strides)):
            f_s, kernel_num, stride = tup
            if i == 0:
                self.model.add(Conv2D(kernel_num, f_s, strides=stride, activation=None, input_shape=(100, 100, 1), data_format="channels_last"))
            else:
                self.model.add(Conv2D(kernel_num, f_s, strides=stride, activation=None))
            #self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last'))

        # flatten (relu) -> fully connected layer
        self.model.add(Flatten())

        # output center_x, center_y, diameter
        self.model.add(Dense(50, activation="relu"))
        self.model.add(Dense(20, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(3, activation="relu"))

        self.model.compile(#loss=tf.losses.huber_loss,
                           loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.Adam())
                           #optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                           #metrics=["mse"])
        print(self.model.summary())

    def fit(self, X_train, Y_train, X_eval, Y_eval, history, epochs):
        return self.model.fit(X_train,
                              Y_train,
                              batch_size=self.batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_data=(X_eval, Y_eval),
                              #callbacks = [history, keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=2e-5, patience=5, verbose=1)])
                              callbacks = [history])

    def predict(self, x, batch_size):
        return self.model.predict(x, batch_size)

    def preprocess(self, x):
        x = x/3
        idx = None
        if len(x.shape) != 2:
            idx = x[:, :, :] < 0.7
        else:
            idx = x[:, :] < 0.7
        x[idx] = 0
        return x

class MSEHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.mse = []

    def on_epoch_end(self, batch, logs={}):
        self.mse.append(logs.get("val_loss"))

def train_on_whole_data():
    K = 5
    k = 0
    epochs = 150
    random.seed(14)
    #fn_X = "X_train_small.npy" # TODO: use args
    #fn_Y = "Y_train_small.npy" # TODO: use args
    #fn_X = "X_train.npy" # TODO: use args
    #fn_Y = "Y_train.npy" # TODO: use args
    fn_X = "X_train_large.npy" # TODO: use args
    fn_Y = "Y_train_large.npy" # TODO: use args
    Xs = np.load(fn_X)
    Ys = np.load(fn_Y)
    print("Xs shape:", Xs.shape)
    print("Ys shape:", Ys.shape, Ys[0])

    # data normalization
    Ys = Ys/200

    # add one dummy channel for X
    Xs = Xs[:,:,:, None]
    print("Xs transformed shape:", Xs.shape)

    # split k-fold: train/eval
    X_train, Y_train, X_eval, Y_eval = split_k_folds(Xs, Ys, k, K)
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_eval shape:", X_eval.shape)
    print("Y_eval shape:", Y_eval.shape)

    # Initialize model with hyper param
    m = Model(filter_sizes = [3, 3, 2, 2],
              kernel_nums = [10, 10, 10, 10],
              strides = [1, 1, 1, 1],
              batch_size = 50,
              training_mode = True)
    
    csv_logger = keras.callbacks.CSVLogger('output.txt')
    
    # train on trainset
    log_history = m.fit(X_train, Y_train, X_eval, Y_eval, csv_logger, epochs)

    # TODO: output metric & hyper params

    ckpt_path = "ckpt/model-{epoch:04d}.ckpt"
    ckpt_dir = os.path.dirname(ckpt_path)
    m.model.save_weights(ckpt_path.format(epoch=epochs))

if __name__ == "__main__":
    train_on_whole_data()

