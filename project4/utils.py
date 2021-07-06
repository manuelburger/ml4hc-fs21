from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix, average_precision_score, roc_auc_score)

from cf_matrix import make_confusion_matrix



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


def compute_metrics(y_true, y_pred):

    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {
        'f1-weighted': f1_weighted,
        'f1-macro': f1_macro,
        'f1-micro': f1_micro,
        'accuracy': acc,
        'balanced_accuracy': bal_acc
    }


def display_result(labels_true, labels_pred, curve=False, proba=None):

    '''
    simple function to display confusion matrix, print f1-score and balanced accuracy
    '''
    if curve and proba is None:
        print("Please provide probalities when computing curve scores!")
        return

    make_confusion_matrix(confusion_matrix(labels_true, labels_pred))
    print(f"{'f1 score micro: ':<25} {f1_score(labels_true, labels_pred, average='micro'):.5}")
    print(f"{'f1 score macro: ':<25} {f1_score(labels_true, labels_pred, average='macro'):.5}")
    print(f"{'f1 score weighted: ':<25} {f1_score(labels_true, labels_pred, average='weighted'):.5}")
    print(f"{'accuracy score: ':<25} {accuracy_score(labels_true, labels_pred):.5}")
    print(f"{'balanced accuracy score: ':<25} {balanced_accuracy_score(labels_true, labels_pred):.5}")
    
    if curve:
        print(f"{'AuPR score: ':<25} {average_precision_score(labels_true, proba):.5}")
        print(f"{'AuROC score: ':<25} {roc_auc_score(labels_true, proba):.5}")
        

def f1_metric(average):

    def fn(y_true, y_pred):
        return f1_score(y_true, np.argmax(y_pred, axis=-1), average=average)
    fn.__name__ = f'f1-{average}'

    return fn


def bac_metric():

    def fn(y_true, y_pred):
        return balanced_accuracy_score(y_true, np.argmax(y_pred, axis=-1))
    fn.__name__ = f'balanced_accuracy'

    return fn


def save_predictions(model, X, model_prefix, data_prefix, nclass=5):

    probabilistic_predictions = model.predict(X, batch_size=512)
    predictions = np.round(probabilistic_predictions) if nclass == 5 \
        else np.argmax(probabilistic_predictions, axis=-1)

    np.save(
        path.join('predictions', f'{data_prefix}_proba_{model_prefix}.npy'),
        probabilistic_predictions,
        allow_pickle=True)

    np.save(
        path.join('predictions', f'{data_prefix}_{model_prefix}.npy'),
        predictions,
        allow_pickle=True)

    # df = pd.DataFrame(
    #     {
    #         'Class probabilities': probabilistic_predictions.tolist(),
    #         'Predicted class': predictions
    #     }
    # )
    #
    # df.to_csv(path.join(
    #     'predictions', model_prefix, f'{data_prefix}_predictions.csv'))


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


def hb_plot(ax, data, **kw):
    """
    Plot mean heartbeats plus/minus one standard deviation
    """
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est,**kw)
    ax.margins(x=0)


def add_channels(X, y, neighbors=1, njobs=8):
    """
    concat samples to get multichannel version
    """
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)
    
    assert X.shape[0] == y.shape[0]
    
    n_channels = 2*neighbors+1
    m, n = X.shape[0], X.shape[1]
    
    out_X = np.zeros((m, n , n_channels))
    out_y = np.zeros((m, n_channels))


    for i in range(m):
        for j in range(n_channels):
            i_clamp = clamp(i+j-neighbors, 0, m-1)
            out_X[i,:,j] = X[i_clamp].flatten()
            out_y[i,j] = Y[i_clamp]
    
    return out_X, out_y


def plot_multiple_heartbeats(X, Y, n_cols=None):

    """
      Input: X: array of signals
             Y: class labels
      Output: multiple plots of heartbeats
    """
    n_cols = n_cols or len(X)
    n_rows = (len(X) - 1) // n_cols + 1
    plt.figure(figsize=(5*n_cols, 3*n_rows))
    
    for index, hb in enumerate(X):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.plot(hb)
        plt.title('class '+ str(Y[index]), fontsize=20)
        # plt.axis("off")
        
    # plt.subplots_adjust(top=1.05)
    plt.tight_layout()
    plt.show()