from os import path

import numpy as np

from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, BatchNormalization)
from keras.callbacks import ModelCheckpoint

from sklearn.utils.class_weight import compute_class_weight

from utils import pred_results
from data_io import get_normalized_data
from data_representation import vectorize_average

work_with_abstracts = True

# We first read in the data that is already normalized,
# but split by the individual abstracts
X_train_full, y_train_full, cy_train_full = get_normalized_data(
    'train', categorical=True, return_abstracts=work_with_abstracts)
X_val_full, y_val_full, cy_val_full = get_normalized_data(
    'val', categorical=True, return_abstracts=work_with_abstracts)
X_test_full, y_test_full, cy_test_full = get_normalized_data(
    'test', categorical=True, return_abstracts=work_with_abstracts)

N_CLASSES = 5

context_sentences = 3
context_type = 'concat'

# We use an embedding model with 300-dimensional embeddings,
# 7 word context size, limited to 32000 most frequent terms and trained for
# 10 epochs with the skip-gram objective
# Although the performance was not affected by the hyperparameters too much,
# this was found to provide stable results
model_name = 'w2v_model_vs300_w7_e10_mc8_lr0.025_sg1_n16_mvs_32000.model'
w2v_model_name = path.join('models', model_name)


# We then preprocess the data further so that the averaged embeddings
# of the individual subsequent sentences are concatenated (if they originate
# from the same abstract). This gives more contextual information.
X_train, y_train, cy_train = vectorize_average(
    X_train_full, y_train_full, cy_train_full,
    context_sentences=context_sentences, context_type=context_type,
    w2v_model_name=w2v_model_name, abstracts=work_with_abstracts)
X_val, y_val, cy_val = vectorize_average(
    X_val_full, y_val_full, cy_val_full,
    context_sentences=context_sentences, context_type=context_type,
    w2v_model_name=w2v_model_name, abstracts=work_with_abstracts)
X_test, y_test, cy_test = vectorize_average(
    X_test_full, y_test_full, cy_test_full,
    context_sentences=context_sentences, context_type=context_type,
    w2v_model_name=w2v_model_name, abstracts=work_with_abstracts)

# We compute the class weights to re-weight the training samples
# This combats the class imbalance in the dataset
cw = compute_class_weight(
    y=y_train, classes=np.arange(5), class_weight='balanced')
class_weights = dict(zip(np.arange(5), cw))


##########################################
# MODEL DEFINITION
##########################################

model = Sequential()

model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(N_CLASSES, activation='softmax'))
print(model.summary())

checkpoint_callback = ModelCheckpoint(
    filepath='dan_models',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

BATCH_SIZE = 16
EPOCHS = 16

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
    class_weight=class_weights,
    verbose=1)


##########################################
# MODEL EVALUATION
##########################################

_, pred_train, cm_train, results_train = pred_results(
    model,
    X_train, y_train,
    model_name='DAN',
    model_type='keras',
    plot_cf=False)

print('Confusion matrix of the training set:')
print(cm_train)
print('Results on the training set:')
print(results_train)

_, pred_val, cm_val, results_val = pred_results(
    model,
    X_val, y_val,
    model_name='DAN',
    model_type='keras',
    plot_cf=False)

print('Confusion matrix of the validation set:')
print(cm_val)
print('Results on the validation set:')
print(results_val)

_, pred_test, cm_test, results_test = pred_results(
    model,
    X_test, y_test,
    model_name='DAN',
    model_type='keras',
    plot_cf=False)

print('Confusion matrix of the test set:')
print(cm_test)
print('Results on the test set:')
print(results_test)

model_name = 'deep_averaging_network'

np.save(f"./predictions/mlp/train_{model_name}.npy", pred_train, allow_pickle=True)
np.save(f"./predictions/mlp/val_{model_name}.npy", pred_val, allow_pickle=True)
np.save(f"./predictions/mlp/test_{model_name}.npy", pred_test, allow_pickle=True)
