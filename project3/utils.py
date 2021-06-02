from string import punctuation, digits
from os import path

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix)

from cf_matrix import make_confusion_matrix


N_CLASSES = 5

# The text labels are encoded as integers
label_map = {
    'BACKGROUND': 0,
    'CONCLUSIONS': 1,
    'METHODS': 2,
    'OBJECTIVE': 3,
    'RESULTS': 4,
}

# The classes we use for text normalisation in the function 'normalize'
# We remove punctuation
translator_punctuation = str.maketrans('', '', punctuation)
# We remove digits
translator_digits = str.maketrans('', '', digits)
stemmer = SnowballStemmer('english')
# We remove stop words
stop_list = set(stopwords.words('english'))

# drugs = pd.read_csv(
#     path.join('data', 'lists', 'general_lists', 'drugs.txt'),
#     header=None)
# medical_terms = pd.read_csv(
#     path.join('data', 'lists', 'general_lists', 'medical_terms.txt'),
#     header=None)
# proper_names = pd.read_csv(
#     path.join('data', 'lists', 'general_lists', 'proper_names.txt'),
#     header=None)
#
# drugs_set = set(drugs[0])
# medical_terms_set = set(
#   [str(term).lower() for term in list(medical_terms[0])])
# proper_names_set = set(proper_names[0])


def normalize(doc, stem=False, return_string=False):
    """
    Input doc and return clean list of tokens
    """
    doc = doc.replace('\r', ' ').replace('\n', ' ')
    lower = doc.lower()
    doc = lower.translate(translator_punctuation)
    doc = doc.split()
    doc = [w for w in doc if w not in stop_list]
    doc = [w for w in doc if not w.startswith('http')]
    doc = [w if not w.isdigit() else '#' for w in doc]

    # We experimented with encoding certain terms
    # (medical, drugs, common names) with special tokens, but that
    # resulted in worse performance

    # doc = [w if w not in drugs_set else '_D' for w in doc]
    # doc = [w if w not in medical_terms_set else '_M' for w in doc]
    # doc = [w if w not in proper_names_set else '_N' for w in doc]

    doc = [w.translate(translator_digits) for w in doc]
    if stem:
        doc = [stemmer.stem(w) for w in doc]
    if return_string:
        doc = " ".join(doc)
    return doc


def pred_results(clf, X, y, model_name=None, plot_cf=False, model_type='keras'):
    """
    Applies classifier clf to test data X, y
    Returns predictions,probabilities and metrics: Balanced Accuracy, F1-score
    Plots the confusion matrix if
    """

    if model_type == 'keras':
        y_proba = clf.predict(X)
    elif model_type == 'xgb':
        y_proba = clf.predict(X, output_margin=False)
    else:
        y_proba = clf.predict_proba(X)

    y_pred = np.argmax(y_proba, axis=1)

    f1_weighted = f1_score(y, y_pred, average='weighted')
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro')
    bal_acc = balanced_accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    results = {
        'model': model_name,
        'f1-weighted': f1_weighted,
        'f1-macro': f1_macro,
        'f1-micro': f1_micro,
        'accuracy': acc,
        'balanced_accuracy': bal_acc
    }

    if plot_cf:
        make_confusion_matrix(
            cm,
            categories=[str(ii) for ii in range(5)],
            cmap='Blues')

    pd.set_option('display.max_columns', None)

    return y_proba, y_pred, cm, pd.DataFrame(results, index=[0])


def preprocess_fasttext(mode='train', normalized=False, labels=True):

    """
    preprocessing for the use with fasttext
    In particular, if labels=True lines will have the form "__label__aaa xxxxxxxxx" as required by fasttext supervised_learning algorithm

    Parameter:
    --------------------------
    mode (str): dataset mode: train, val or test
    normalized (bool): whether sentences will be normalized
    labels (bool): whether to include "__label__" for supervised training

    Output:
    --------------------------
    saves txt-file ready for training resp. predictions with fasttext

    """

    file_name = ''
    if mode == 'train':
        file_name = path.join('data', 'train.txt')
    elif mode == 'val':
        file_name = path.join('data', 'val.txt')
    elif mode == 'test':
        file_name = path.join('data', 'test.txt')

    data_out = []

    with open(file_name, 'r') as f_input:
        for line in f_input:

            if line.startswith('#') or line == '\n':
                continue
            label, text = line.split('\t')[:2]

            if normalized:
                text = normalize(text, return_string=True)

            if labels:
              sentence_out = '__label__'+label+' '+ text[:-1]
            else:
              sentence_out = text[:-1]
            data_out.append(sentence_out)

    with open('data/fasttext{norm}_{mode}{label}.txt'.format(norm='_norm' if normalized else '', mode=mode, label='_wo_lbls' if (not labels) and (mode=='train') else '' ), 'w') as filehandle:
      for sentences in data_out:
          filehandle.write('%s\n' % sentences)


def pred_fasttext(model, normalized=False, mode='test'):

    '''
    Input (model):
    ------------------------
    Trained fasttext model

    Parameter
    -----------------------
    normalized (bool): whether to use normalized sentences or not
    mode (bool): mode of dataset: train, val, test

    Output:
    -----------------------
    pred (list): list of predictions of the model for respective dataset
    '''

    file_name = './data/fasttext{norm}_{mode}.txt'.format(norm='_norm' if normalized else '', mode=mode)
    pred = []

    with open(file_name, 'r') as f_input:
        for line in f_input:

          pred.append(model.predict(line.strip('\n'))[0][0].strip('__label__'))

    return pred


def display_result(labels_true, labels_pred):

    '''
    simple function to display confusion matrix, print f1-score and balanced accuracy
    '''

    make_confusion_matrix(confusion_matrix(labels_true, labels_pred))
    print(f"{'f1 score micro: ':<25} {f1_score(labels_true, labels_pred, average='micro'):.5}")
    print(f"{'f1 score weighted: ':<25} {f1_score(labels_true, labels_pred, average='weighted'):.5}")
    print(f"{'balanced accuracy score: ':<25} {balanced_accuracy_score(labels_true, labels_pred):.5}")


def compute_metrics_callback(pred):
    """
    A callback function for the Huggingface package training validation.
    """
    y = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    f1_weighted = f1_score(y, y_pred, average='weighted')
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro')
    bal_acc = balanced_accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return {
        'f1-weighted': f1_weighted,
        'f1-macro': f1_macro,
        'f1-micro': f1_micro,
        'accuracy': acc,
        'balanced_accuracy': bal_acc
    }


class MeanEmbeddingVectorizer(object):
    
    '''
    class to get mean embeddings based on dictionary {word:embedded vector}
    '''
    
    def __init__(self, dict):
        self.dict = dict
        if len(dict)>0:
            self.dim=len(next(iter(dict.values())))
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.dict[w] for w in doc if w in self.dict] 
                    or [np.zeros(self.dim)], axis=0)
            for doc in X
        ])