# Author: Hamaad Musharaf Shah.

import math
import inspect

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, local, convolutional, Flatten, Reshape
from keras.models import Model

import tensorflow

from autoencoders_keras.loss_history import LossHistory

class ConvolutionalAutoencoder(BaseEstimator, 
                               TransformerMixin):
    def __init__(self, 
                 input_shape=None,
                 n_epoch=None,
                 batch_size=None,
                 encoder_layers=None,
                 decoder_layers=None,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 pool_size=None,
                 denoising=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        
        for arg, val in values.items():
            setattr(self, arg, val)
        
        loss_history = LossHistory()
        self.callbacks_list = [loss_history]
        
        for i in range(self.encoder_layers):
            if i == 0:
                with tensorflow.device("/gpu:0"):
                    self.input_data = Input(shape=self.input_shape)
                    self.encoded = BatchNormalization()(self.input_data)
                    self.encoded = keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.encoded)
                    self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i > 0 and i < self.encoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.encoded = BatchNormalization()(self.encoded)
                    self.encoded = keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.encoded)
                    self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i == self.encoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.encoded = BatchNormalization()(self.encoded)
                    self.encoded = keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.encoded)
                    self.encoded = Dropout(rate=0.5)(self.encoded)

        with tensorflow.device("/gpu:0"):
            self.encoded = keras.layers.MaxPooling1D(strides=self.pool_size, padding="valid")(self.encoded)
            self.encoded = BatchNormalization()(self.encoded)
            self.encoded = keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.encoded)
            self.decoded = keras.layers.UpSampling1D(size=self.pool_size)(self.encoded)

        for i in range(self.decoder_layers):
            if i == 0:
                with tensorflow.device("/gpu:0"):
                    self.decoded = BatchNormalization()(self.decoded)
                    self.decoded = keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.decoded)
                    self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i > 0 and i < self.decoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.decoded = BatchNormalization()(self.decoded)
                    self.decoded = keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.decoded)
                    self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i == self.decoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.decoded = BatchNormalization()(self.decoded)
                    self.decoded = keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.decoded)
                    self.decoded = Dropout(rate=0.5)(self.decoded)

        with tensorflow.device("/gpu:0"):
            # 3D tensor with shape: (batch_size, new_steps, filters).
            # Remember think of this as a 2D-Lattice per observation.
            # Rows represent time and columns represent some quantities of interest that evolve over time.
            self.decoded = BatchNormalization()(self.decoded)
            self.decoded = keras.layers.Conv1D(filters=self.input_shape[1], kernel_size=self.kernel_size, strides=self.strides, activation="sigmoid", padding="same")(self.decoded)

            self.autoencoder = Model(self.input_data, self.decoded)
            self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                     loss="mean_squared_error")
            
    def fit(self,
            X,
            y=None):
        with tensorflow.device("/gpu:0"):
            keras.backend.get_session().run(tensorflow.global_variables_initializer())
            self.autoencoder.fit(X if self.denoising is None else X + self.denoising, X,
                                 validation_split=0.05,
                                 epochs=self.n_epoch,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 callbacks=self.callbacks_list, 
                                 verbose=1)

            self.encoder = Model(self.input_data, self.encoded)
        
        return self
    
    def transform(self,
                  X):
        with tensorflow.device("/gpu:0"):
            out = np.reshape(self.encoder.predict(X), (X.shape[0], self.filters * int(X.shape[1] / self.pool_size)))
            
        return out