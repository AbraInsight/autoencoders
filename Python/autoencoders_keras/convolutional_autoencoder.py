# Author: Hamaad Musharaf Shah.

import math
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, local, convolutional, pooling, Flatten, Reshape
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
                 encoding_dim=None,
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
                    self.encoded = convolutional.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.encoded)
                    self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i > 0 and i < self.encoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.encoded = BatchNormalization()(self.encoded)
                    self.encoded = convolutional.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.encoded)
                    self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i == self.encoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.encoded = BatchNormalization()(self.encoded)
                    self.encoded = convolutional.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.encoded)
                    self.encoded = Dropout(rate=0.5)(self.encoded)
                    self.encoded = pooling.MaxPooling1D(self.pool_size, padding="same")(self.encoded)

        with tensorflow.device("/gpu:0"):
            self.encoded = BatchNormalization()(self.encoded)
            self.encoded = local.LocallyConnected1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="valid")(self.encoded)
            self.encoded = Flatten()(self.encoded)
            self.encoded = BatchNormalization()(self.encoded)
            self.encoded = Dense(self.encoding_dim, activation="sigmoid")(self.encoded)
            self.decoded = BatchNormalization()(self.encoded)
            self.decoded = Dense(int(input_shape[1] / self.pool_size) * self.encoding_dim, activation="elu")(self.decoded)

        for i in range(self.decoder_layers):
            if i == 0:
                with tensorflow.device("/gpu:0"):
                    self.decoded = convolutional.UpSampling1D(self.pool_size)(Reshape((int(input_shape[1] / self.pool_size), self.encoding_dim))(self.decoded))
                    self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i > 0 and i < self.decoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.decoded = BatchNormalization()(self.decoded)
                    self.decoded = convolutional.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.decoded)
                    self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i == self.decoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.decoded = BatchNormalization()(self.decoded)
                    self.decoded = convolutional.Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.decoded)
                    self.decoded = Dropout(rate=0.5)(self.decoded)
        
        with tensorflow.device("/gpu:0"):
            # 3D tensor with shape: (batch_size, new_steps, filters).
            # Remember think of this as a 2D-Lattice per observation.
            # Rows represent time and columns represent some quantities of interest that evolve over time.
            self.decoded = BatchNormalization()(self.decoded)
            self.decoded = convolutional.Conv1D(filters=self.input_shape[1], kernel_size=self.kernel_size, strides=self.strides, activation="sigmoid", padding="same")(self.decoded)

            self.autoencoder = Model(self.input_data, self.decoded)
            self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                     loss="mean_squared_error")
            
    def fit(self,
            X,
            y=None):
        with tensorflow.device("/gpu:0"):
            keras.backend.get_session().run(tensorflow.global_variables_initializer())
            self.autoencoder.fit(X if self.denoising is None else X + self.denoising, X,
                                 validation_split=0.3,
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
            out = self.encoder.predict(X)
            
        return out