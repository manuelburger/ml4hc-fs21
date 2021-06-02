import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm import tqdm

from nltk import ngrams

from imblearn.under_sampling import RandomUnderSampler


def get_data(species='worm', form='split', mode='train', k=1, drop=0.0, crop=None):
    """
        Reads in the appropriate data set for the selected species.
        For the human data set, the provided splits were used.
        For the C. elegans data, we split the provided sequences into
        3 data sets: train, validation, and test, with ratios 70:15:15.
        The function can drop a selected portion of negative samples 
        as a simple way of undersampling.
        It also splits the strings into k-mers if form='split' is chosen
        can crops the input around the central region (around the possible splice site)
        if the parameter crop is specified.
    """
    
    data_dir = 'data'
    if species == 'worm':
        file_dict = { 'train' : 'C_elegans_train_split.csv',
                      'val' : 'C_elegans_validation_split.csv',
                      'test'  : 'C_elegans_test_split.csv'} 
        data_file = file_dict[mode]
    elif species == 'human':
        file_dict = { 'train' : 'human_dna_train_split.csv',
                      'val' : 'human_dna_validation_split.csv',
                      'test'  : 'human_dna_test_split.csv',
                      'hidden': 'human_dna_test_hidden_split.csv'}
        data_file = file_dict[mode]

    df = pd.read_csv(os.path.join(data_dir, data_file))
    y = None if mode == "hidden" else df['labels'].to_numpy()
    
    if drop > 0.0:
        # Drop a portion of negative samples
        negative = y == -1
        negative_places = np.nonzero(negative)[0]
        negative[np.random.choice(
            negative_places, replace=False,
            size=int(drop * len(negative_places)))] = False
        positive = y == 1

        to_keep = negative + positive

        df = df.iloc[to_keep]
        y = y[to_keep]

    if crop is not None:
        center = len(df['sequences'][0])//2
        df['sequences'] = [ f[max(0,(center - crop)):min((center + crop),len(f))] for f in df['sequences']]
        
    if form == 'split':
        # Split into k-mers
        df['split_sequences'] = df['sequences'].apply(list)
        
        if k > 1:
            df['split_sequences'] = df['split_sequences'].apply(
                lambda e: [''.join(f) for f in ngrams(e, k)])
        
        no_sequences = len(df['split_sequences'])
        sequence_length = len(df['split_sequences'].iloc[0])
        X = np.zeros(
            shape=(no_sequences, sequence_length), dtype=f'<U{k}')
        for ii in tqdm(range(no_sequences)):
            X[ii, :] = np.array(
                df['split_sequences'].iloc[ii])
    
    else:

        X = df['sequences'].to_list()
    
    return X, y



def balance_out(X, y, method='random', ratio=0.5):
    if method == 'random':
        rus = RandomUnderSampler(sampling_strategy=ratio)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled   
    
    
def get_dataset_shapes(data_dir = "data"):
    ''' get shapes of all the datasets'''
  
    files = os.listdir(data_dir)
    shapes = []

    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        shapes.append([f, df.shape])

    return shapes