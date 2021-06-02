###############################################
# Data representation script
# This script was used to build appropriate representations of the inputs
###############################################

from os import path

import numpy as np
from tqdm.notebook import tqdm
from scipy.signal import convolve2d


s_values = {
    1: 1,
    3: 2,
    5: 2,
    7: 4
}


def vectorize_set(X, w2v_model, embedding_size):
    """
    Vectorizes the list of documents X
    by taking the mean across all dimensions.
    @param X: List of sentences represented as a list of words
              (the whole data set or individual abstracts)
    @param w2v_model: Model to use for embedding individual words
    @param embedding_size: Size of the embedding.
    @return: Vector representation of the sentences and a mask of which
             sentences were kept (some may be discarded
             due to insufficient data
    """
    X_vect = np.zeros(shape=(len(X), embedding_size))
    for ii in range(len(X)):
        x = X[ii]
        a = np.zeros(shape=(len(x), embedding_size))
        for jj, w in enumerate(x):
            if w in w2v_model.wv:
                a[jj, :] = w2v_model.wv[w]
        X_vect[ii, :] = np.mean(a, axis=0)

    X_vect[np.isnan(X_vect).any(axis=1)] = 0
    # X_vect = X_vect[keep]
    keep = ~np.isnan(X_vect).any(axis=1)

    return X_vect, keep


def do_concatenate(X, window_size):
    """
    Concatenates the vector representations of neighboring sentences to
    enlarge the context size. alltogether, ```window_size``` number of
    sentences are concatenated
    @param X: list of vector representations of sentences
             (whole dataset or individual abstracts)
    @param window_size: number of sentences to concatenate together
    @return: List of concatenated sentence representations
    """
    reach = window_size // 2
    n, m = X.shape

    X_tmp = np.zeros(shape=(n, m * window_size))

    for ii in range(-reach, reach):
        X_tmp[reach: -reach, m * (ii + reach): m * ((ii + reach) + 1)] = \
            X[reach + ii: -reach + ii, :]

    for jj in range(0, reach):
        for ii in range(reach - jj):
            X_tmp[jj, m * ii: m * (ii + 1)] = X[0, :]
        for ii in range(reach - jj - 1, window_size):
            X_tmp[jj, m * ii: m * (ii + 1)] = X[min(ii, n - 1), :]

    for jj in range(n - 1, n - reach - 1, -1):
        for ii in range(reach + (n - 1 - jj), window_size):
            X_tmp[jj, m * ii: m * (ii + 1)] = X[-1, :]
        for ii in range(reach + (n - 1 - jj) - 1, 0, -1):
            X_tmp[jj, m * ii: m * (ii + 1)] = X[jj - (reach - ii), :]

    return X_tmp


def flatten(X, y, cy):
    """
    Flattens a list of lists (list ob abstracts) into a flat list of sentences
    @param X: list of abstracts (each is a list of sentences)
    @param y: list of labels per abstract as numeric labels
    @param cy: list of labels per abstract as 1-hot encoded
    @return: flattened inputs
    """
    yy = []
    for l in y:
        yy.extend(list(l))
    cyy = []
    for l in cy:
        cyy.extend(list(l))

    XX = []
    for l in X:
        XX.extend(l)

    return np.asarray(XX), np.asarray(yy), np.asarray(cyy)


def vectorize_average(
        X, y_o, cy_o=None, abstracts=False,
        context_sentences=1, context_type='average', w2v_model_name=None):
    """
    Embeds each sentence as the average of the representations of its words.
    It also groups the neighboring sentences for a larger context.
    @param X: The original (preprocessed) sentences
    @param y_o: The original numerical labels
    @param cy_o: The original 1-hot-encoded labels
    @param abstracts: Whether X contains abstracts (lists of sentences) or
           sentences (lists of words)
    @param context_sentences: Number of sentences to concatenate
           (size of context)
    @param context_type: Whether to average neighboring sentences ('average')
           or concatenate them ('concat')
    @param w2v_model_name: Name of the Word2Vec model to use to get the
           vector representations
    @return: Encoded sentences and the labels that were kept (some may be
             removed due to insufficient data)
    """
    import gensim.models
    w2v_model = gensim.models.Word2Vec.load(w2v_model_name)

    y = y_o.copy()
    cy = cy_o.copy()
    embedding_size = int(w2v_model_name.split('_')[2][2:])

    if not abstracts:

        X_vect, keep = vectorize_set(X, w2v_model, embedding_size)

        y = y[keep]
        cy = cy[keep]

    else:
        X_vect = []
        for ii, abstract in enumerate(X):
            abstract_vect, abstract_keep = vectorize_set(
                abstract, w2v_model, embedding_size)
            X_vect.append(abstract_vect)

            y[ii] = y[ii][abstract_keep]
            cy[ii] = cy[ii][abstract_keep]

    if context_type == 'average':

        s = s_values[context_sentences]
        a1 = np.arange(1, (context_sentences // 2) * s + 2, s)
        a2 = np.arange((context_sentences // 2 - 1) * s + 1, 0, -s)
        kernel = np.concatenate((a1, a2)).reshape((-1, 1))
        kernel = kernel / np.sum(kernel)

        if not abstracts:
            X_vect = convolve2d(X_vect, kernel, mode='same', boundary='symm')
        else:
            for ii in len(X_vect):
                X_vect[ii] = convolve2d(X_vect[ii], kernel, mode='same', boundary='symm')

    elif context_type == 'concat':

        if not abstracts:
            X_vect = do_concatenate(X_vect, window_size=context_sentences)
        else:
            for ii in range(len(X_vect)):
                X_vect[ii] = do_concatenate(
                    X_vect[ii], window_size=context_sentences)

            X_vect, y, cy = flatten(X_vect, y, cy)

    if cy is not None:
        return X_vect, y, cy
    else:
        return X_vect, y


def do_concatenate_sentences(X, window_size, normalized=True):
    """
    Concatenates sentences (represented as a string or list of words
    - not vectors) into a larger context.
    @param X: List of sentences (as lists of words if normalized=True
              or 1 string if normalized=False)
    @param window_size: Size of context
    @param normalized: Whether the sentences are normalized (preprocessed and
           represented as a list of words)
    @return: List of concatenated sentences
    """
    reach = window_size // 2
    n = len(X)

    X_cat = []
    for ii in range(n):
        if normalized:
            context = []
            for jj in range(max(0, ii - reach), min(n, ii + reach + 1)):
                context.extend(X[jj])
        else:
            context = ''
            for jj in range(max(0, ii - reach), min(n, ii + reach + 1)):
                context += X[jj][0]
        X_cat.append(context)

    return X_cat


def prepare_sequential_data(X, y, cy, window_size=3, normalized=True):
    """
    Helper function that prepares the raw (preprocessed/normalized) data
    to be used in a RNN model.
    @param X: List of sentences/abstracts
    @param y: The original numerical labels
    @param cy: The original 1-hot-encoded labels
    @param window_size: Size of context
    @param normalized: Whether the data is normalized
           (sentences split into lists of words)
    @return:
    """

    X_tmp = []
    for x in X:
        X_tmp.append(do_concatenate_sentences(
            x, window_size=window_size, normalized=normalized))

    return flatten(X_tmp, y, cy)
