import datetime
from os import path

import numpy as np

import gensim.models

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (
    Dense, Dropout, Input, Embedding, LSTM, Bidirectional, BatchNormalization)
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from data_io import get_normalized_data
from utils import N_CLASSES, pred_results
from data_representation import prepare_sequential_data


model_name = 'w2v_model_vs300_w7_e10_mc8_lr0.025_sg1_n16_mvs_32000.model'

w2v_model = gensim.models.Word2Vec.load(path.join('models', model_name))

embedding_size = int(model_name.split('_')[2][2:])
work_with_abstracts = True
window_size = 3
VOCAB_SIZE = len(w2v_model.wv.key_to_index)
MAX_SEQUENCE_LENGTH = 96
BATCH_SIZE = 128
EPOCHS = 4


# We first read in the data that is already normalized,
# but split by the individual abstracts
X_train_full, y_train_full, cy_train_full = get_normalized_data(
    'train', categorical=True, return_abstracts=work_with_abstracts)
X_val_full, y_val_full, cy_val_full = get_normalized_data(
    'val', categorical=True, return_abstracts=work_with_abstracts)
X_test_full, y_test_full, cy_test_full = get_normalized_data(
    'test', categorical=True, return_abstracts=work_with_abstracts)

# We then prepare the data to be used by the RNN. Sentences are concatenated
# and represented as a list of words.
X_train_full, y_train_full, cy_train_full = prepare_sequential_data(
    X_train_full, y_train_full, cy_train_full, window_size=window_size)
X_val_full, y_val_full, cy_val_full = prepare_sequential_data(
    X_val_full, y_val_full, cy_val_full, window_size=window_size)
X_test_full, y_test_full, cy_test_full = prepare_sequential_data(
    X_test_full, y_test_full, cy_test_full, window_size=window_size)

# The dictionary that maps each word to the 1-hot encoding id
word_index = {
    key: w2v_model.wv.key_to_index[key] + 1
    for key in w2v_model.wv.key_to_index}

# We encode the sentences as a sequence of token ids
train_sequences = [
    [word_index.get(t, 0) for t in text] for text in X_train_full]
val_sequences = [
    [word_index.get(t, 0) for t in text] for text in X_val_full]
test_sequences = [
    [word_index.get(t, 0) for t in text] for text in X_test_full]

# We pad and truncate the sequences to the maximum length which was chosen
# empirically based on the distribution of lengths of concatenated inputs.
X_train = pad_sequences(
    train_sequences, maxlen=MAX_SEQUENCE_LENGTH,
    padding='post', truncating='post')
X_val = pad_sequences(
    val_sequences, maxlen=MAX_SEQUENCE_LENGTH,
    padding='post', truncating='post')
X_test = pad_sequences(
    test_sequences, maxlen=MAX_SEQUENCE_LENGTH,
    padding='post', truncating='post')

# We initialize the matrix of word embeddings
wv_matrix = (np.random.rand(VOCAB_SIZE + 1, embedding_size) - 0.5) / 5.0
# And fill it with values of the embeddings
for word, i in word_index.items():
    wv_matrix[i, :] = w2v_model.wv[word]


##########################################
# MODEL DEFINITION
##########################################
model = Sequential()

model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
model.add(Embedding(
    VOCAB_SIZE + 1, embedding_size,
    weights=[wv_matrix],
    trainable=False))
model.add(Bidirectional(LSTM(embedding_size, dropout=0.4, recurrent_dropout=0.25)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(N_CLASSES, activation='softmax'))

print(model.summary())

checkpoint_callback = ModelCheckpoint(
    filepath='w2v_rnn_models',
    save_weights_only=True,
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
    y=cy_train_full,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, cy_val_full),
    callbacks=[checkpoint_callback],
    verbose=1)

##########################################
# MODEL EVALUATION
##########################################

_, pred_train, cm_train, results_train = pred_results(
    model,
    X_train, y_train_full,
    model_name='w2vRNN',
    model_type='keras',
    plot_cf=False)

print('Confusion matrix of the training set:')
print(cm_train)
print('Results on the training set:')
print(results_train)

_, pred_val, cm_val, results_val = pred_results(
    model,
    X_val, y_val_full,
    model_name='w2vRNN',
    model_type='keras',
    plot_cf=False)

print('Confusion matrix of the validation set:')
print(cm_val)
print('Results on the validation set:')
print(results_val)

_, pred_test, cm_test, results_test = pred_results(
    model,
    X_test, y_test_full,
    model_name='w2vRNN',
    model_type='keras',
    plot_cf=False)

print('Confusion matrix of the test set:')
print(cm_test)
print('Results on the test set:')
print(results_test)

model_name = 'w2v_rnn'

np.save(f"./predictions/w2v_rnn/train_{model_name}.npy", pred_train, allow_pickle=True)
np.save(f"./predictions/w2v_rnn/val_{model_name}.npy", pred_val, allow_pickle=True)
np.save(f"./predictions/w2v_rnn/test_{model_name}.npy", pred_test, allow_pickle=True)


