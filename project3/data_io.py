from os import path

import numpy as np

from utils import label_map, normalize


def to_categorical(y, num_classes=None, dtype='float32'):
    """
    Transforms numerically-encoded labels into 1-hot encoded ones
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def read_data(mode, return_abstracts=False, concatenate_sentences=False):
    """
    Modified data reading function. It directly reads in the data
    """
    file_name = ''
    if mode == 'train':
        file_name = path.join('data', 'train.txt')
    elif mode == 'val':
        file_name = path.join('data', 'val.txt')
    elif mode == 'test':
        file_name = path.join('data', 'test.txt')

    text_data = []
    target_data = []

    concatenated_sentences = []

    abstract_n = 0
    abstract_index_data = []
    abstract_text = []
    abstract_texts = []
    abstract_target = []
    abstract_targets = []

    with open(file_name, 'r') as f_input:
        for line in f_input:
            if line.startswith('#') or line == '\n':
                if line.startswith('#'):
                    if abstract_n > 0:
                        abstract_texts.append(abstract_text)
                        abstract_targets.append(abstract_target)
                    abstract_n += 1
                    concatenated_sentences.append('')
                    abstract_text = []
                    abstract_target = []
                continue
            target, text = line.split('\t')[:2]

            concatenated_sentences[-1] += (' ' + text[:-1])

            abstract_index_data.append(abstract_n)
            text_data.append(text[:-1])
            abstract_text.append(text[:-1])
            abstract_target.append(target)
            target_data.append(target)

    abstract_texts.append(abstract_text)
    abstract_targets.append(abstract_target)

    if return_abstracts:
        if concatenate_sentences:
            return text_data, target_data, abstract_texts, abstract_targets, concatenated_sentences
        else:
            return text_data, target_data, abstract_texts, abstract_targets
    else:
        if concatenate_sentences:
            return text_data, target_data, concatenated_sentences
        else:
            return text_data, target_data


def get_normalized_data(mode, categorical=False, stem=False,
                        return_abstracts=False, concatenate_sentences=False,
                        normalized=True):
    """
    A helper function to directly read in (un)normalized data by our custom
    normalization function.
    @param mode: 'train', 'val', or 'test'
    @param categorical: Whether to include categorically-encoded labels
    @param stem: Whether to stem the words
    @param return_abstracts: Whether to return the individual abstract as lists
           of sentences
    @param concatenate_sentences: Whether to return the abstracts as
           concatenated sentences
    @param normalized: Whether to normalize sentences (and represent them as
           lists of words) or leave them as raw strings.
    @return: Processed data
    """
    if return_abstracts:
        if concatenate_sentences:
            texts, labels_, abstract_texts, abstract_targets, concatenated_sentences = \
                read_data(mode, return_abstracts, concatenate_sentences)
        else:
            texts, labels_, abstract_texts, abstract_targets = read_data(
                mode, return_abstracts, concatenate_sentences)
    else:
        if concatenate_sentences:
            texts, labels_, concatenated_sentences = read_data(
                mode, return_abstracts, concatenate_sentences)
        else:
            texts, labels_ = read_data(
                mode, return_abstracts, concatenate_sentences)

    if not return_abstracts:
        y = np.asarray([label_map[label] for label in labels_])
        cy = to_categorical(y, num_classes=5)
        ds = texts if not concatenate_sentences else concatenated_sentences
        if normalized:
            normalized_text = np.array(
                [normalize(doc, stem) for doc in ds], dtype='object')
        else:
            normalized_text = np.array(ds, dtype='object')
    else:
        y, cy = [], []
        for at in abstract_targets:
            y.append(np.asarray([label_map[label] for label in at]))
            cy.append(to_categorical(y[-1], num_classes=5))
        if normalized:
            normalized_text = np.array(
                [[normalize(doc, stem) for doc in a] for a in abstract_texts],
                dtype='object')
        else:
            normalized_text = [[[doc] for doc in a] for a in abstract_texts]

    if categorical:
        return normalized_text, y, cy
    else:
        return normalized_text, y
