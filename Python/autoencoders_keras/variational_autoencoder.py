# Author: Hamaad Musharaf Shah.

import math
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda
from keras.models import Model, Sequential

import tensorflow

from autoencoders_keras.loss_history import LossHistory

class VariationalAutoencoder(BaseEstimator, 
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
            self.mu = Dense(units=self.encoding_dim, activation="linear")(self.encoded)
            self.log_sigma = Dense(units=self.encoding_dim, activation="linear")(self.encoded)
            z = Lambda(self.sample_z, output_shape=(self.encoding_dim,))([self.mu, self.log_sigma])

        self.decoded_layers_dict = {}
        for i in range(self.decoder_layers):
            with tensorflow.device("/gpu:0"):
                if i == 0:
                    self.decoded_layers_dict[i] = Dense(units=self.n_hidden_units, activation="elu")
                    self.decoded = self.decoded_layers_dict[i](z)
                elif i > 0 and i < self.decoder_layers - 1:
                    self.decoded_layers_dict[i] = Dense(units=self.n_hidden_units, activation="elu")
                    self.decoded = self.decoded_layers_dict[i](self.decoded)
                elif i == self.decoder_layers - 1:
                    self.decoded_layers_dict[i] = Dense(units=self.n_hidden_units, activation="elu")
                    self.decoded = self.decoded_layers_dict[i](self.decoded)
        
        with tensorflow.device("/gpu:0"):
            # Output would have shape: (batch_size, n_feat).
            self.decoded_layers_dict[self.decoder_layers] = Dense(units=self.n_feat, activation="sigmoid")
            self.decoded = self.decoded_layers_dict[self.decoder_layers](self.decoded)
        
            self.autoencoder = Model(self.input_data, self.decoded)
            self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                     loss=self.vae_loss)
            
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

            self.encoder = Model(self.input_data, self.mu)

            self.generator_input = Input(shape=(self.encoding_dim,))
            self.generator_output = None
            for i in range(self.decoder_layers):
                if i == 0:
                    self.generator_output = self.decoded_layers_dict[i](self.generator_input)
                elif i > 0 and i < self.decoder_layers - 1:
                    self.generator_output = self.decoded_layers_dict[i](self.generator_output)
                elif i == self.decoder_layers - 1:
                    self.generator_output = self.decoded_layers_dict[i](self.generator_output)

            self.generator_output = self.decoded_layers_dict[self.decoder_layers](self.generator_output)

            self.generator = Model(self.generator_input, self.generator_output)
                
        return self
    
    def transform(self,
                  X):
        with tensorflow.device("/gpu:0"):
            out = self.encoder.predict(X)
            
        return out
    
    def sample_z(self,
                 args):
        with tensorflow.device("/gpu:0"):
            mu_, log_sigma_ = args
            eps = keras.backend.random_normal(shape=(keras.backend.shape(mu_)[0], self.encoding_dim),
                                              mean=0.0,
                                              stddev=1.0)
            out = mu_ + keras.backend.exp(log_sigma_ / 2) * eps
            
        return out
    
    def vae_loss(self,
                 y_true,
                 y_pred):
        with tensorflow.device("/gpu:0"):
            recon = self.n_feat * keras.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
            kl = -0.5 * keras.backend.mean(1.0 + self.log_sigma - keras.backend.exp(self.log_sigma) - keras.backend.square(self.mu), axis=-1)
            out = recon + kl
            
        return out