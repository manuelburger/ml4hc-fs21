import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from cf_matrix import make_confusion_matrix
from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score, precision_recall_curve, roc_curve, roc_auc_score,  auc)



def get_class_distribution(y):
    """
        A helper function that returns the number of instances of each class
    """
    unique, counts = np.unique(y, return_counts=True)
    return {-1: counts[0], 1: counts[1]}


def plot_class_distribution(y):
    """
        A helper function that plots the number of instances of each class
    """
    sns.barplot(
        data=pd.DataFrame.from_dict(
            [get_class_distribution(y)]).melt(), 
            x='variable', y='value', hue='variable').set_title(
                'Class Distribution in the Data Set')


def binarize(y, pos=1, neg=-1, thresh=0.5):
    y_pred = y.copy()
    y_pred[y_pred < thresh] = neg
    y_pred[y_pred >= thresh] = pos
    return y_pred

 
def pred_results(clf, X, y, model_name=None, figsize=None, model_type='generic', threshold=0.5):
    
    '''
    Appplies classifier clf to test data X, y
    Returns predictions,probabilities and metrics: Balanced Accuracy, F1-score, ROCAUC and PRCAUC
    Plots ROC and PR-curve when figsize not None
    
    '''
    
    if model_type == 'keras':
        y_proba = clf.predict(X)
        y_pred = [0 if y < threshold else 1 for y in y_proba]
    elif model_type == 'xgb':
        y_proba = clf.predict(X, output_margin=False)
        y_pred = [0 if y < threshold else 1 for y in y_proba]
    else:
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)

    f1 = f1_score(y, y_pred, average='micro')
    bal_acc = balanced_accuracy_score(y, y_pred)
    
    if model_type in ['keras', 'xgb']:
        fpr, tpr, tresh_roc = roc_curve(y.ravel(), y_proba.ravel())
        precision, recall, tresh_roc = precision_recall_curve(y.ravel(), y_proba.ravel())
    else:
        fpr, tpr, tresh_roc = roc_curve(y.ravel(), y_proba[:,1].ravel())
        precision, recall, tresh_roc = precision_recall_curve(y.ravel(), y_proba[:,1].ravel())
        
    roc_auc = auc(fpr, tpr)
    prc_auc = auc(recall, precision) 
  
    results = {
        'model' : model_name,  
        'f1-score' : f1,
        'balanced_accuracy' : bal_acc,
        'roc_auc' : roc_auc,
        'prc_auc' : prc_auc
        }
    
    if figsize is not None: 
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    
        make_confusion_matrix(confusion_matrix(y, y_pred), 
        group_names=[
            'True Negative',
            'False Positive',
            'False Negative',
            'True Positive'],
        categories=['-1', '1'],
        cmap='Blues',
        ax = axes[0])
    
        sns.lineplot(ax=axes[1], x=fpr, y=tpr)
        axes[1].set(xlabel='False Positive Rate (1-specificity)' + '\n\nROC AUC={:0.3f}'.format(roc_auc), ylabel='True Positive Rate (Recall)', title='ROC Curve')
    
        sns.lineplot(ax=axes[2], x=recall, y=precision)
        axes[2].set(xlabel='Recall'+ '\n\nPRC AUC={:0.3f}'.format(prc_auc), ylabel='Precision' , title='PR Curve')

    return y_proba, y_pred, pd.DataFrame(results, index=[0])


def compute_scores(y, y_proba, model_name=None, figsize=None, threshold=0.5):
    
    '''
    Appplies classifier clf to test data X, y
    Returns predictions,probabilities and metrics: Balanced Accuracy, F1-score, ROCAUC and PRCAUC
    Plots ROC and PR-curve when figsize not None
    
    '''
    
    y_pred = [0 if y < threshold else 1 for y in y_proba]

    f1 = f1_score(y, y_pred, average='micro')
    bal_acc = balanced_accuracy_score(y, y_pred)
    
    fpr, tpr, tresh_roc = roc_curve(y, y_proba)
    precision, recall, tresh_roc = precision_recall_curve(y, y_proba)
        
    roc_auc = auc(fpr, tpr)
    prc_auc = auc(recall, precision) 
  
    results = {
        'model' : model_name,  
        'f1-score' : f1,
        'balanced_accuracy' : bal_acc,
        'roc_auc' : roc_auc,
        'prc_auc' : prc_auc
        }
    
    if figsize is not None: 
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    
        make_confusion_matrix(confusion_matrix(y, y_pred), 
        group_names=[
            'True Negative',
            'False Positive',
            'False Negative',
            'True Positive'],
        categories=['-1', '1'],
        cmap='Blues',
        ax = axes[0])
    
        sns.lineplot(ax=axes[1], x=fpr, y=tpr)
        axes[1].set(xlabel='False Positive Rate (1-specificity)' + '\n\nROC AUC={:0.3f}'.format(roc_auc), ylabel='True Positive Rate (Recall)', title='ROC Curve')
    
        sns.lineplot(ax=axes[2], x=recall, y=precision)
        axes[2].set(xlabel='Recall'+ '\n\nPRC AUC={:0.3f}'.format(prc_auc), ylabel='Precision' , title='PR Curve')

    return y_proba, y_pred, pd.DataFrame(results, index=[0])