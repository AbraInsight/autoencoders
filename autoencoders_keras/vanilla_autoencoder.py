# Author: Hamaad Musharaf Shah.
# The following references were used.
# https://keras.io
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://stackoverflow.com/questions/42177658/how-to-switch-backend-with-keras-from-tensorflow-to-theano
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
# http://scikit-learn.org/stable/
# Book: Ian Goodfellow, Yoshua Bengio and Aaron Courville, "Deep Learning" - http://www.deeplearningbook.org
# Book: Aurelien Geron, "Hands-On Machine Learning with Scikit-Learn & Tensorflow" - https://www.amazon.co.uk/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291

import math
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, regularizers
from keras.models import Model

from autoencoders_keras.loss_history import LossHistory

class VanillaAutoencoder(BaseEstimator, TransformerMixin):
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

        self.input_data = Input(shape=(self.n_feat,))
        
        for i in range(self.encoder_layers):
            if i == 0:
                self.encoded = Dense(self.n_hidden_units, activation="elu")(self.input_data)
                self.encoded = BatchNormalization()(self.encoded)
                self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i > 0 and i < self.encoder_layers - 1:
                self.encoded = Dense(self.n_hidden_units, activation="elu")(self.encoded)
                self.encoded = BatchNormalization()(self.encoded)
                self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i == self.encoder_layers - 1:
                self.encoded = Dense(self.n_hidden_units, activation="elu")(self.encoded)
                self.encoded = BatchNormalization()(self.encoded)
        
        self.encoded = Dense(self.encoding_dim, activation="sigmoid", activity_regularizer=regularizers.l1(1e-4))(self.encoded)

        for i in range(self.decoder_layers):
            if i == 0:
                self.decoded = Dense(self.n_hidden_units, activation="elu")(BatchNormalization()(self.encoded))
                self.decoded = BatchNormalization()(self.decoded)
                self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i > 0 and i < self.decoder_layers - 1:
                self.decoded = Dense(self.n_hidden_units, activation="elu")(self.decoded)
                self.decoded = BatchNormalization()(self.decoded)
                self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i == self.decoder_layers - 1:
                self.decoded = Dense(self.n_hidden_units, activation="elu")(self.decoded)
                self.decoded = BatchNormalization()(self.decoded)
        
        # Output would have shape: (batch_size, n_feat).
        self.decoded = Dense(self.n_feat, activation="sigmoid")(self.decoded)
        
        self.autoencoder = Model(self.input_data, self.decoded)
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                 loss="mean_squared_error")
    def fit(self,
            X,
            y=None):
        self.autoencoder.fit(X if self.denoising is None else X + self.denoising, X,
                             validation_split=0.3,
                             epochs=self.n_epoch,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=self.callbacks_list, 
                             verbose=2)
        
        self.encoder = Model(self.input_data, self.encoded)
        
        return self
    
    def transform(self,
                  X):
        return self.encoder.predict(X)                 