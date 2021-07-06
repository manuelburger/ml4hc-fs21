import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def _get_data(mode, dataset):
    if dataset == 'mitbih':
        df_train = pd.read_csv(f"./data/mitbih_{mode}.csv", header=None)
        df_train = df_train.sample(frac=1, random_state=42)

        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        return X, Y
    else:

        df_1 = pd.read_csv("./data/ptbdb_normal.csv", header=None)
        df_2 = pd.read_csv("./data/ptbdb_abnormal.csv", header=None)
        df = pd.concat([df_1, df_2])

        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=1337, stratify=df[187])

        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        Y_test = np.array(df_test[187].values).astype(np.int8)
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

        return X, Y, X_test, Y_test


def get_data(dataset):
    if dataset == 'mitbih':
        X, y = _get_data('train', dataset)
        X_test, y_test = _get_data('test', dataset)

        return X, y, X_test, y_test
    else:

        return _get_data('', dataset)
