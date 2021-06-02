import os

import numpy as np

import tensorflow as tf

from transformers import (
    TFBertForSequenceClassification, BertTokenizerFast,
    TFTrainer, TFTrainingArguments)

from data_io import get_normalized_data
from utils import N_CLASSES, pred_results, compute_metrics_callback
from data_representation import prepare_sequential_data


# We use a model which was pretrained on full texts from PubMed
MODEL_NAME = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_SHORT = 'camb_SapBERT'

MAX_LENGTH = 96
BATCH_SIZE = 32
WINDOW_SIZE = 3

# We load in the data which is not normalized by our custom function.
# The preprocessing is left to the implemented tokenizer.
# The sentences are grouped by abstracts so that they can be
# correctly concatenated in the next step.
normalized = False
X_train_full, y_train_full, cy_train_full = get_normalized_data(
    'train', categorical=True, return_abstracts=True, normalized=normalized)
X_val_full, y_val_full, cy_val_full = get_normalized_data(
    'val', categorical=True, return_abstracts=True, normalized=normalized)
X_test_full, y_test_full, cy_test_full = get_normalized_data(
    'test', categorical=True, return_abstracts=True, normalized=normalized)

# We concatenate the sentences with their 3-sentence context that occurs
# in the same abstract.
X_train, y_train, cy_train = prepare_sequential_data(
    X_train_full, y_train_full, cy_train_full,
    window_size=WINDOW_SIZE, normalized=normalized)
X_val, y_val, cy_val = prepare_sequential_data(
    X_val_full, y_val_full, cy_val_full,
    window_size=WINDOW_SIZE, normalized=normalized)
X_test, y_test, cy_test = prepare_sequential_data(
    X_test_full, y_test_full, cy_test_full,
    window_size=WINDOW_SIZE, normalized=normalized)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# Tokenization of the inputs
train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True, padding=True,
    max_length=MAX_LENGTH,
    is_split_into_words=normalized)
val_encodings = tokenizer(
    X_val.tolist(),
    truncation=True, padding=True,
    max_length=MAX_LENGTH,
    is_split_into_words=normalized)
test_encodings = tokenizer(
    X_test.tolist(),
    truncation=True, padding=True,
    max_length=MAX_LENGTH,
    is_split_into_words=normalized)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

##########################################
# MODEL DEFINITION
##########################################

run_name = 'ml4hc-p2-tf-{}-ws{}-bs{}-ml{}-ep2-default'.format(
    MODEL_NAME_SHORT, WINDOW_SIZE, BATCH_SIZE, MAX_LENGTH)

training_args = TFTrainingArguments(
    output_dir='./tf-bert-results',
    num_train_epochs=2,  # We fine-tune for only 2 epochs
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    warmup_steps=0,
    weight_decay=0.01,  # Regularization
    report_to='wandb',
    logging_dir='./tf-bert-logs',
    logging_steps=10,
    run_name=run_name,
    learning_rate=5e-5,
    evaluation_strategy='steps',
    eval_steps=10000
)

with training_args.strategy.scope():
    model = TFBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES)


##########################################
# MODEL TRAINING
##########################################

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_callback
)

trainer.train()

os.makedirs('bert_tf_models', exist_ok=True)
model.save_pretrained(os.path.join('bert_tf_models', 'latest_model_2e.model'))


##########################################
# MODEL EVALUATION & RESULTS
##########################################

model_name = 'bert_hf_pubmed_keras'

pred_train = trainer.predict(train_dataset)
pred_val = trainer.predict(val_dataset)
pred_test = trainer.predict(test_dataset)

preds_train = np.argmax(pred_train.predictions, axis=1)
preds_val = np.argmax(pred_val.predictions, axis=1)
preds_test = np.argmax(pred_test.predictions, axis=1)

model_name = 'bert_keras'

np.save(f"./predictions/bert/train_{model_name}.npy", preds_train, allow_pickle=True)
np.save(f"./predictions/bert/val_{model_name}.npy", preds_val, allow_pickle=True)
np.save(f"./predictions/bert/test_{model_name}.npy", preds_test, allow_pickle=True)
