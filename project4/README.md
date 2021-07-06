# ECG time series data classification

> Manuel Burger, Tobias Peter, Anej Svete

## Content:

The submission contains the following files:

1. Juypter notebooks to train the various models:

    - `baselines.ipynb`: Implementation and reproduction of the baselines 
    - `baseline_improvements.ipynb`: Implementation of two basic extensions (CNN, RNN) of the baseline.
    - `residual_cnn.ipynb`: Implementation of the CNN with residual layers
    - `attention_rnn.ipynb`: Implementation of an RNN with an attention layer (not in the report)
    - `ensembles.ipynb`: Code to read in predictions of other models and combine them by averaging or with logistic regression
    - `additional_cnn.ipynb`: CNN with dilated convolutions, Gaussian noise layers and residual connection with extracted features.
    - `bilstm.ipynb`: Implementation of a Bi-directional LSTM
    

2. Python scripts
	- Utilities:
		- `create_directories.py`: Creates necessary folder structure
		- `model_parts.py`: definitions of various models and their parts (residual blocks etc.), training function used for CNN with dilated convolutions
		- `cf_matrix.py`: custom confusion matrix
		- `data.py`: helper functions for feature extraction
		- `data_io.py`: helper functions for data input/output
		- `utils.py`: helper functions


3. `ml4hc_project4_env.yml`
	 - yml-file defining the conda environment
	 

4. `ML4HC_Project_4_Report.pdf`
	 - Project report describing the used models and their results