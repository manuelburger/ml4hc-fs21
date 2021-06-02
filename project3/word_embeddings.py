###############################################
# Word2Vec script
# This script was used to build embedding models for the abstracts with a
# grid search over multiple hyperparameters
###############################################

from os import path
import logging

from sklearn.model_selection import ParameterGrid

import gensim.models

from data_io import get_normalized_data

# This command enables more detailed logging of progress
# logging.basicConfig(
#     format='%(asctime)s : %(levelname)s : %(message)s',
#     level=logging.INFO)

# We train the model on training data alone. We Obtain the normalized
# sentences in the form of a list
normalized_text_train, _ = get_normalized_data(
    'train', stem=False, concatenate_sentences=False)

model_type = 'w2v'
model = gensim.models.Word2Vec

# Setting the grid of hyperparameters
# The optimal values were chosen extrinsically (based on the performance
# of the model using them).
parameters = {
    'embedding_size': [300],
    'window_length': [7],
    'epochs': [10],
    'min_count': [8],
    'sg': [1],
    'negative': [16],
    'max_vocab_size': [32000],
}

parameter_grid = ParameterGrid(parameters)

# Building and saving the model with each set of hyperparameters
for ii, p in enumerate(parameter_grid):

    print(f'On iteration {ii + 1}/{len(parameter_grid)}.')

    EMBEDDING_SIZE = p['embedding_size'] if 'embedding_size' in p else 100
    WINDOW_LENGTH = p['window_length'] if 'window_length' in p else 5
    EPOCHS = p['epochs'] if 'epochs' in p else 10
    MIN_COUNT = p['min_count'] if 'min_count' in p else 5
    LEARNING_RATE = p['learning_rate'] if 'learning_rate' in p else 2.5e-2
    SKIP_GRAM = p['sg'] if 'sg' in p else 1
    NEGATIVE = p['negative'] if 'negative' in p else 5
    MAX_VOCAB_SIZE = p['max_vocab_size'] if 'max_vocab_size' in p else None

    embedder = model(
        sentences=normalized_text_train,
        vector_size=EMBEDDING_SIZE,
        window=WINDOW_LENGTH,
        epochs=EPOCHS,
        min_count=MIN_COUNT,
        alpha=LEARNING_RATE,
        sg=SKIP_GRAM,
        negative=NEGATIVE,
        max_vocab_size=MAX_VOCAB_SIZE,
        workers=16,
    )

    embedder.save(path.join(
        'models',
        'w2v_model_vs{}_w{}_e{}_mc{}_lr{}_sg{}_n{}_mvs_{}_A.model'.format(
            EMBEDDING_SIZE, WINDOW_LENGTH, EPOCHS, MIN_COUNT,
            LEARNING_RATE, SKIP_GRAM, NEGATIVE, MAX_VOCAB_SIZE)))
