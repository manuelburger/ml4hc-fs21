from typing import overload
import itertools

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from nltk import ngrams

from tqdm import tqdm

from glove import Glove

    
def one_hot_encoding(X, form='1d', k=1):
    """
        1-hot encodes each k-mer in the list of sequences X.
        The embedding can be 1-dimensional, where each binary feature 
        (presence of a specific kmer at a specific position)
        has its own column (for use with most ML models)
        or 2-dimentional where each position is encoded with 
        a 1-hot vector (for use with 1-d convolutions).
    """
    categories = list(
        map(''.join, itertools.product('ATCG', repeat=k)))
    enc = OneHotEncoder(
        drop='first',
        handle_unknown='error',
        categories=[categories] * X.shape[1])
    if form == '1d':
        return enc.fit_transform(X)
    else:
        channels = len(categories) - 1
        X = enc.fit_transform(X).todense()
        Z = np.zeros((X.shape[0], X.shape[1] // channels, channels))
        for ii in range(X.shape[0]):
            Z[ii, :, :] = np.reshape(X[ii, :], (channels, -1), order='F').T
        return Z


def numeric_encoding(X):
    
    mapper = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'T': 0.1335
    }
    
    Z = [[mapper[c] for c in list(r)] for r in X]
   
    return np.array(Z)
    

def kmer_embeddings(X, k, to_split=False):
    """
        Uses the trained GloVe model to return the dense encodings of the overlapping k-mers 
        in the list of input sequences X.
        Each input sequence is encoded as a (l, d) matrix, where l is the length of the sequence
        and d is the number of embedding dimensions
    """
    
    glove = Glove.load('glove-w24-d30-e100.model')
    
    if to_split:
        X = [[''.join(f) for f in ngrams(e, k)] for e in X]
    
    D = np.zeros(shape=(len(X), len(X[0]), glove.no_components), dtype=np.float32)
    
    for ii, x in tqdm(enumerate(X), total = len(X)):
        D[ii, :, :] = np.array([
            glove.word_vectors[glove.dictionary[kmer]]
            if kmer in glove.dictionary else np.zeros(shape=(glove.no_components,)) for kmer in x])
        
    return D
    

def kmer_counts(X, k=3, init_kmers=None):  ## TODO: init_kmers...
    kmer_freqs = []
    if init_kmers is None:
        kmers = set()
    for x in tqdm(X):
        freq = x.kmer_frequencies(k, overlap=True)
        kmer_freqs.append(freq)
        if init_kmers is None:
            kmers.update(freq.keys())
    kmers = list(sorted(kmers)) if init_kmers is None else init_kmers
    F = np.zeros(shape=(len(X), len(kmers)))
    for ii in tqdm(range(len(X))):
        for jj in range(len(kmers)):
            F[ii, jj] = kmer_freqs[ii].get(kmers[jj], 0.0)
    return F


def gram_matrix(X, kernel):
    G = np.zeros(shape=(len(X), len(X)))
    for ii in tqdm(range(len(X))):
        for jj in range(len(X)):
            G[ii, jj] = kernel(X[ii], X[jj])
    return G
