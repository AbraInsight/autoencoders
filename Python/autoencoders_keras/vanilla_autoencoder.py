# Author: Hamaad Musharaf Shah.

import math
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model

import tensorflow

from autoencoders_keras.loss_history import LossHistory

class VanillaAutoencoder(BaseEstimator, 
                         TransformerMixin):
    def __init__(self, 
                 n_feat=None,
                 n_epoch=None,
                 batch_size=None,
                 encoder_layers=None,
                 decoder_layers=None,
                 n_hidden_units=None,
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
                    self.input_data = Input(shape=(self.n_feat,))
                    self.encoded = BatchNormalization()(self.input_data)
                    self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)
                    self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i > 0 and i < self.encoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.encoded = BatchNormalization()(self.encoded)
                    self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)
                    self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i == self.encoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.encoded = BatchNormalization()(self.encoded)
                    self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)
        
        with tensorflow.device("/gpu:0"):
            self.encoded = BatchNormalization()(self.encoded)
            self.encoded = Dense(units=self.encoding_dim, activation="sigmoid")(self.encoded)

        for i in range(self.decoder_layers):
            if i == 0:
                with tensorflow.device("/gpu:0"):
                    self.decoded = BatchNormalization()(self.encoded)
                    self.decoded = Dense(units=self.n_hidden_units, activation="elu")(self.decoded)
                    self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i > 0 and i < self.decoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.decoded = BatchNormalization()(self.decoded)
                    self.decoded = Dense(units=self.n_hidden_units, activation="elu")(self.decoded)
                    self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i == self.decoder_layers - 1:
                with tensorflow.device("/gpu:0"):
                    self.decoded = BatchNormalization()(self.decoded)
                    self.decoded = Dense(units=self.n_hidden_units, activation="elu")(self.decoded)
        
        with tensorflow.device("/gpu:0"):
            # Output would have shape: (batch_size, n_feat).
            self.decoded = BatchNormalization()(self.decoded)
            self.decoded = Dense(units=self.n_feat, activation="sigmoid")(self.decoded)
        
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
            out = self.encoder.predict(X)
            
        return out