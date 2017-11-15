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
from keras.layers import Input, Dense, BatchNormalization, Dropout, regularizers, local, convolutional, pooling, Flatten, Reshape, LSTM, RepeatVector
from keras.models import Model

from autoencoders_keras.loss_history import LossHistory

class Seq2SeqAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 input_shape=None,
                 n_epoch=None,
                 batch_size=None,
                 encoder_layers=None,
                 decoder_layers=None,
                 n_hidden_units=None,
                 encoding_dim=None,
                 stateful=None,
                 unroll=None,
                 denoising=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        
        for arg, val in values.items():
            setattr(self, arg, val)
        
        loss_history = LossHistory()
        self.callbacks_list = [loss_history]
        
        # 2D-lattice with time on the x-axis (across rows) and with space on the y-axis (across columns).
        if self.stateful is True:
            self.input_data = Input(batch_shape=self.input_shape)
            self.n_rows = self.input_shape[1]
            self.n_cols = self.input_shape[2]
        else:
            self.input_data = Input(shape=self.input_shape)
            self.n_rows = self.input_shape[0]
            self.n_cols = self.input_shape[1]

        for i in range(self.encoder_layers):
            if i == 0:
                # Returns n_rows sequences of vectors of dimension encoding_dim.
                self.encoded = LSTM(units=self.n_hidden_units, activation="elu", return_sequences=True, stateful=self.stateful, unroll=self.unroll)(self.input_data)
                self.encoded = BatchNormalization()(self.encoded)
                self.encoded = Dropout(rate=0.5)(self.encoded)
            else:
                self.encoded = LSTM(units=self.n_hidden_units, activation="elu", return_sequences=True, stateful=self.stateful, unroll=self.unroll)(self.encoded)
                self.encoded = BatchNormalization()(self.encoded)
                self.encoded = Dropout(rate=0.5)(self.encoded)

        # Returns 1 vector of dimension encoding_dim.
        self.encoded = LSTM(units=self.encoding_dim, activation="sigmoid", return_sequences=False, stateful=self.stateful, unroll=self.unroll)(self.encoded)
        self.decoded = RepeatVector(self.n_rows)(BatchNormalization()(self.encoded))

        for i in range(self.decoder_layers):
            self.decoded = LSTM(units=self.n_hidden_units, activation="elu", return_sequences=True, stateful=self.stateful, unroll=self.unroll)(self.decoded)
            self.decoded = BatchNormalization()(self.decoded)
            self.decoded = Dropout(rate=0.5)(self.decoded)
        
        # If return_sequences is True: 3D tensor with shape (batch_size, timesteps, units).
        # Else: 2D tensor with shape (batch_size, units).
        # If return_state is True: a list of tensors. 
        # The first tensor is the output. The remaining tensors are the last states, each with shape (batch_size, units).
        # If stateful is True: the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
        # If unroll is True: the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.
        self.decoded = LSTM(units=self.n_cols, activation="sigmoid", return_sequences=True, stateful=self.stateful, unroll=self.unroll)(self.decoded)
        
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