import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import *
from config import *
import numpy as np

X_train = np.genfromtxt("x_train.csv",delimiter=",", dtype=float)
y_train = np.genfromtxt("y_train.csv",delimiter=",", dtype=float)
X_test = np.genfromtxt("x_test.csv",delimiter=",", dtype=float)
y_test = np.genfromtxt("y_test.csv",delimiter=",", dtype=float)

output_size = X_train.shape[1]
input_size = (X_train.shape[1],)

def train_model(candidate):
    latent_size = candidate[-1]
    input_layers = candidate[:4]
    output_layers = candidate[4:-1]

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        encoder = Sequential()
        encoder.add(InputLayer(input_shape=input_size))
        for nodes in input_layers:
            encoder.add(Dense(nodes, activation='tanh'))
        encoder.add(Dense(latent_size, activation='tanh'))
        
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(latent_size,)))
        for nodes in output_layers:
            decoder.add(Dense(nodes, activation='tanh'))
        decoder.add(Dense(output_size, activation='tanh'))

        inputs = Input(shape = input_size)
        latent_vector = encoder(inputs)
        output = decoder(latent_vector)

        model = Model(inputs = inputs, outputs = output)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = keras.losses.MeanAbsolutePercentageError()
        metrics = ['accuracy']
        model.compile(optimizer, metrics = metrics, loss = loss)

    batch_size = batch_size * strategy.num_replicas_in_sync
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    train_loss = history.history['loss'][-1]
    train_accuracy = history.history['accuracy'][-1]
    return train_loss

def latent_count(candidate):
    return candidate[-1]

def param_count():
    pass