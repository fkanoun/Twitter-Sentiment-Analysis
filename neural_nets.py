################# Relevant imports #################

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, GRU, SpatialDropout1D, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

# DataFrame manipulation
import pandas as pd

# Matrix manipulation
import numpy as np

from preprocessing_helpers import create_embedding_matrix

import sys

################# Create the models #################


def get_model_by_name(algo_name, X, y, serie, vocabulary_size, max_length):
    """
    Returns the model and the history for the chosen algo name

    INPUT:
        algo_name : string        - The name of the algo chosen
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained and the history of the training
    """
    if (algo_name == 'embeddings'):
        model, history = build_model_embeddings(serie, y)
    elif (algo_name == 'lstm'):
        model, history = build_model_lstm(X, y, vocabulary_size, max_length)
    elif (algo_name == 'cnn_lstm'):
        model, history = build_model_cnn_lstm(X, y, vocabulary_size, max_length)
    elif (algo_name == 'bidir_gru'):
        model, history = build_model_bidir_gru(X, y, vocabulary_size, max_length)
    elif (algo_name == 'cnn_gru'):
        model, history = build_model_cnn_gru(X, y, vocabulary_size, max_length)
    else :
        print('Enter the correct name')
        print('program stopping')
        sys.exit()
    return model, history


def build_model_lstm(X,
                     y,
                     vocabulary_size,
                     max_length,
                     callbacks_list=[
                         ModelCheckpoint(
                             filepath='LSTM_best_weights.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max'),
                         EarlyStopping(
                             monitor='val_acc', patience=3, mode='max')
                     ],
                     Embedding_size=200,
                     batch_size=16384,
                     validation_split=0.04,
                     epochs=100):
    """
    Create the model for a Long Short-Term Memory Network

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results
        callbacks_list :          - The callback options for the model
        Embedding_size            - The size of the embedding
        batch_size                - The size of the batch in the neural network
        validation_split          - The validation_test split
        epochs                    - The number of epochs


    OUTPUT:
        Returns the model trained and the history of the training
    """
    print('Using Long Short-Term Memory Network')

    model_lstm = Sequential()
    model_lstm.add(
        Embedding(vocabulary_size, Embedding_size, input_length=max_length))
    model_lstm.add(LSTM(Embedding_size, dropout=0.2, recurrent_dropout=0.2))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_lstm.summary()

    history = model_lstm.fit(
        X,
        y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list)

    return model_lstm, history


def build_model_cnn_lstm(X,
                         y,
                         vocabulary_size,
                         max_length,
                         callbacks_list=[
                             ModelCheckpoint(
                                 filepath='CNN_LSTM_best_weights.hdf5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max'),
                             EarlyStopping(
                                 monitor='val_acc', patience=3, mode='max')
                         ],
                         Embedding_size=200,
                         batch_size=16384,
                         validation_split=0.04,
                         epochs=100):
    """
    Create the model for a Convolutional Neural Network with a Long Short-Term Memory Network

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results
        callbacks_list :          - The callback options for the model
        Embedding_size            - The size of the embedding
        batch_size                - The size of the batch in the neural network
        validation_split          - The validation_test split
        epochs                    - The number of epochs


    OUTPUT:
        Returns the model trained and the history of the training
    """
    print('Using Convolutional Neural Network with a Long Short-Term Memory Network')

    model_conv = Sequential()
    model_conv.add(
        Embedding(vocabulary_size, Embedding_size, input_length=max_length))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(Embedding_size))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_conv.summary()

    history_conv = model_conv.fit(
        X,
        y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list)

    return model_conv, history_conv


def build_model_cnn_gru(X,
                        y,
                        vocabulary_size,
                        max_length,
                        callbacks_list=[
                            ModelCheckpoint(
                                filepath='CNN_GRU_best_weights.hdf5',
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max'),
                            EarlyStopping(
                                monitor='val_acc', patience=3, mode='max')
                        ],
                        Embedding_size=200,
                        batch_size=16384,
                        validation_split=0.04,
                        epochs=100):
    """
    Create the model for a Convolutional Neural Network with Gated Recurrent Unit

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results
        callbacks_list :          - The callback options for the model
        Embedding_size            - The size of the embedding
        batch_size                - The size of the batch in the neural network
        validation_split          - The validation_test split
        epochs                    - The number of epochs


    OUTPUT:
        Returns the model trained and the history of the training
    """
    print('Using  Convolutional Neural Network with Gated Recurrent Unit')

    model_gru_cnn = Sequential()
    model_gru_cnn.add(
        Embedding(vocabulary_size, Embedding_size, input_length=max_length))
    model_gru_cnn.add(
        Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model_gru_cnn.add(MaxPooling1D(pool_size=2))
    model_gru_cnn.add(Dropout(0.25))
    model_gru_cnn.add(GRU(128, return_sequences=True))
    model_gru_cnn.add(Dropout(0.3))
    model_gru_cnn.add(Flatten())
    model_gru_cnn.add(Dense(128, activation='relu'))
    model_gru_cnn.add(Dropout(0.5))
    model_gru_cnn.add(Dense(1, activation='sigmoid'))
    model_gru_cnn.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy'])

    model_gru_cnn.summary()

    history_gru_cnn = model_gru_cnn.fit(
        X,
        y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list)

    return model_gru_cnn, history_gru_cnn


def build_model_bidir_gru(X,
                          y,
                          vocabulary_size,
                          max_length,
                          callbacks_list=[
                              ModelCheckpoint(
                                  filepath='BIDIR_GRU_best_weights.hdf5',
                                  monitor='val_acc',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='max'),
                              EarlyStopping(
                                  monitor='val_acc', patience=3, mode='max')
                          ],
                          Embedding_size=200,
                          batch_size=16384,
                          validation_split=0.04,
                          epochs=100):
    """
    Create the model for a Neural Network with Bidirectional Gated Recurrent Unit

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results
        callbacks_list :          - The callback options for the model
        Embedding_size            - The size of the embedding
        batch_size                - The size of the batch in the neural network
        validation_split          - The validation_test split
        epochs                    - The number of epochs


    OUTPUT:
        Returns the model trained and the history of the training
    """
    print('Using Neural Network with Bidirectional Gated Recurrent Unit')

    model_Bidir_GRU = Sequential()

    model_Bidir_GRU.add(
        Embedding(vocabulary_size, Embedding_size, input_length=max_length))
    model_Bidir_GRU.add(SpatialDropout1D(0.25))
    model_Bidir_GRU.add(Bidirectional(GRU(128)))
    model_Bidir_GRU.add(Dropout(0.5))

    model_Bidir_GRU.add(Dense(1, activation='sigmoid'))
    model_Bidir_GRU.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy'])

    model_Bidir_GRU.summary()

    history_Bidir_GRU = model_Bidir_GRU.fit(
        X,
        y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list)

    return model_Bidir_GRU, history_Bidir_GRU


def build_model_embeddings(X,
                           y,
                           batch_size_=200,
                           validation_split_=0.2,
                           epochs_=100,
                           callbacks_list=[
                               ModelCheckpoint(
                                   filepath='Embeddings_best_weights.hdf5',
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max'),
                               EarlyStopping(
                                   monitor='val_acc', patience=3, mode='max')
                           ]):
    """
    Create the model for a Neural Network based on the word embeddings of a word2vec model.

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results
        batch_size_                - The size of the batch in the neural network
        validation_split_          - The validation_test split
        epochs_                - The number of epochs


    OUTPUT:
        Returns the model trained and the history of the training
    """

    print('Neural Network based on the word embeddings of a word2vec model')

    vector_dimension, vocabulary_size, embedding_matrix, data,tokenizer = create_embedding_matrix(
        X, pretrained=False)
    #Change pre-trained to true if you want to use the pre-trained glove model

    model = Sequential()
    model.add(
        Embedding(
            vocabulary_size,
            vector_dimension,
            input_length=50,
            weights=[embedding_matrix],
            trainable=False))
    model.add(
        Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(
        Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(
        Conv1D(filters=32, kernel_size=7, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(
        Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=["acc"])
    model.summary()

    history = model.fit(
        data,
        y,
        batch_size=batch_size_,
        verbose=1,
        validation_split=validation_split_,
        epochs=epochs_,
        callbacks=callbacks_list)

    return model, history


def predict(model, X_test,tokenizer, max_length, X, algo_name = None):
    """
    Create a list with predictions corresponding to the models

    INPUT:
        model       - The trained model
        X_test      - DataFrame that has the tweets for which we want to predict the sentiments

    OUTPUT:
        Returns a list of 0s and 1s corresponding to the sentiments
    """
    if (algo_name == 'embeddings'):
        _, _, _, _,tokenizer = create_embedding_matrix(
            X, pretrained=False)
        max_length = 50


    test_sequences = tokenizer.texts_to_sequences(X_test['tweets'])
    test = pad_sequences(test_sequences, maxlen=max_length)

    preds = model.predict(test)
    predictions = []

    for x in preds:
        if (x <= 0.5):
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions
