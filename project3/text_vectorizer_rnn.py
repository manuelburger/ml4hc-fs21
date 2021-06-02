from os import path

import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (
    Dense, Dropout, Input, Embedding, LSTM, Bidirectional, BatchNormalization)
from keras.callbacks import ModelCheckpoint
from keras.layers.experimental.preprocessing import TextVectorization

from data_io import get_normalized_data
from utils import N_CLASSES, pred_results
from data_representation import prepare_sequential_data


##########################################
# DATA
##########################################

work_with_abstracts = True
window_size = 3

# We first read in the data that is *not* normalized (raw sentences),
# but split by the individual abstracts
X_train_full, y_train_full, cy_train_full = get_normalized_data(
    'train', categorical=True, return_abstracts=work_with_abstracts,
    normalized=False)
X_val_full, y_val_full, cy_val_full = get_normalized_data(
    'val', categorical=True, return_abstracts=work_with_abstracts,
    normalized=False)
X_test_full, y_test_full, cy_test_full = get_normalized_data(
    'test', categorical=True, return_abstracts=work_with_abstracts,
    normalized=False)

# We then prepare the data to be used by the RNN.
# Subsequent sentences are concatenated.
X_train, y_train, cy_train = prepare_sequential_data(
    X_train_full, y_train_full, cy_train_full,
    window_size=window_size, normalized=False)
X_val, y_val, cy_val = prepare_sequential_data(
    X_val_full, y_val_full, cy_val_full,
    window_size=window_size, normalized=False)
X_test, y_test, cy_test = prepare_sequential_data(
    X_test_full, y_test_full, cy_test_full,
    window_size=window_size, normalized=False)

VOCAB_SIZE = 32000
MAX_SEQUENCE_LENGTH = 96
EMBEDDING_SIZE = 256

# Since the network is very large (~11 million parameters), we only train
# for 2 epochs and take relatively large bathces
BATCH_SIZE = 128
EPOCHS = 2

##########################################
# MODEL DEFINITION
##########################################

# We define the TextVectorizer layer which takes in raw sentences
# (and their concatenated context) and transforms them into a (padded)
# sequence of tokens of length MAX_SEQUENCE_LENGTH.
# These are then used as inputs to the embedding layer.
vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    ngrams=1,
    output_sequence_length=MAX_SEQUENCE_LENGTH)
vectorize_layer.adapt(X_train)

# We define the model that transforms the raw sentences, encodes them with
# learned randomly initialized embeddings and passes them through an
# two-layer RNN and later through a number of dense layers.
model = Sequential()

model.add(vectorize_layer)
model.add(Input(shape=(1,), dtype=tf.string))
model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, trainable=True))
model.add(Bidirectional(
    LSTM(EMBEDDING_SIZE, dropout=0.4, recurrent_dropout=0.25,
         return_sequences=True)))
model.add(Bidirectional(
    LSTM(EMBEDDING_SIZE, dropout=0.4, recurrent_dropout=0.25)))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(N_CLASSES, activation='softmax'))


checkpoint_callback = ModelCheckpoint(
    filepath='vect_rnn_2lstm_{epoch:02d}_{val_accuracy:.4f}.model',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

##########################################
# MODEL TRAINING
##########################################

history = model.fit(
    x=X_train,
    y=cy_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, cy_val),
    callbacks=[checkpoint_callback],
    verbose=1)

model.save(path.join('vect_rnn_2lstm_models', 'latest_model.model'))

##########################################
# MODEL EVALUATION
##########################################

_, pred_train, _, _ = pred_results(
    model,
    X_train, y_train,
    model_name='VectRNN',
    model_type='keras',
    plot_cf=False)

_, pred_val, _, _ = pred_results(
    model,
    X_val, y_val,
    model_name='VectRNN',
    model_type='keras',
    plot_cf=False)

_, pred_test, _, _ = pred_results(
    model,
    X_test, y_test,
    model_name='VectRNN',
    model_type='keras',
    plot_cf=False)

# Save the predictions
model_name = 'text_vect_2lstm'

np.save(f"./predictions/train_{model_name}.npy", pred_train, allow_pickle=True)
np.save(f"./predictions/val_{model_name}.npy", pred_val, allow_pickle=True)
np.save(f"./predictions/test_{model_name}.npy", pred_test, allow_pickle=True)


