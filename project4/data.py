import multiprocessing
import pandas as pd
from tqdm import tqdm
from functools import partial

import tsfel
from tsfel.feature_extraction.calc_features import calc_window_features


def extract_feature(e, cfg, fs=125.0):
    return calc_window_features(cfg, e, fs=fs)


def get_tsfel_features(train, test, domain="temporal", njobs=8, sampling_rate=125.0):
    '''
    Extract features with the TSFEL library
    
    @param train: training data as an iterable
    @param test: test data as an iterable
    @param domain: spectral, temporal, statistical
    @param njobs: number of workers to use for parallel processing
    '''
    
    cfg_file = tsfel.get_features_by_domain(domain) 
    
    with multiprocessing.Pool(processes=njobs) as p:
        train_f = p.map(partial(extract_feature, cfg=cfg_file, fs=sampling_rate), tqdm(train))
        test_f = p.map(partial(extract_feature, cfg=cfg_file, fs=sampling_rate), tqdm(test))
        
    return pd.concat(train_f), pd.concat(test_f)