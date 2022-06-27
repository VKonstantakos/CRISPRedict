import numpy as np
import pandas as pd
import statsmodels.api as sm
from features import *


def get_model(promoter='u6', classifier=False, path='../Saved models/CRISPRedict'):
  '''
  Loads the appropriate model for the chosen task.

  :parameter promoter: gRNA transcription method to load the suitable model ('u6', 't7').
  :parameter classifier: option to load a classification model. Default is False.
  :parameter path: file path containing the saved models.
  :return: loaded model.
  '''

  # Check selected promoter
  assert promoter in ['u6', 't7'], "No promoter was selected.\
  Select a valid promoter ('u6' or 't7') to load the appropriate model."

  # Load classification model
  if classifier == True:
    model_type = 'classifier'

  # Load regression model
  elif classifier == False:
    model_type = 'regressor'

  else:
    raise ValueError(
        "Select a valid model type (classifier is True or False).")

  model = sm.load(
      f'{path}/CRISPRedict_{promoter}_{model_type}.pickle')

  return model


def predict_sequence(sequence, promoter=None, classifier=False, path='../Saved models/CRISPRedict'):
  '''
  Predict the efficiency of a given sequence.

  :parameter sequence: sequence to analyse.
  :parameter promoter: gRNA transcription method to load the suitable model ('u6', 't7').
  :parameter classifier: option to use a classification model. Default is False.
  :parameter path: file path containing the saved models.
  :return: predicted efficiency or (confidence score, label) in the case of classification.
  '''

  # Check selected promoter
  assert promoter in ['u6', 't7'], "No promoter was selected.\
  Select a valid promoter ('u6' or 't7') to load the appropriate model."

  # Check model type
  assert classifier in [True, False], "Select a valid model type (classifier is True or False)."

  # Define model
  model = get_model(promoter=promoter, classifier=classifier, path=path)

  # Create features and get prediction
  features = seq_features(sequence, single=True, promoter=promoter)
  features = sm.add_constant(features, has_constant='add')
  prediction = model.predict(features)[0]

  if classifier == True:
    label = 1 if prediction >= 0.5 else 0
    return prediction, label

  return prediction


def predict_file(filepath, seq_col='30-nt sequence', promoter=None, classifier=False, path='../Saved models/CRISPRedict', save=True):
  '''
  Predict the efficiency of multiple sequences given in a CSV file.

  :parameter filepath: path to a CSV file that contains 30-nt sequences.
  :parameter seq_col: name of column containing 30-nt sequences.
  :parameter promoter: gRNA transcription method to load the suitable model ('u6', 't7').
  :parameter classifier: option to use a classification model. Default is False.
  :parameter path: file path containing the saved models.
  :parameter save: option to save the model's predictions. Default is True.
  :return: dataframe containing the model's predictions.
  '''

  # Check selected promoter
  assert promoter in ['u6', 't7'], "No promoter was selected.\
  Select a valid promoter ('u6' or 't7') to load the appropriate model."

  # Check model type
  assert classifier in [True, False], "Select a valid model type (classifier is True or False)."

  # Define model
  model = get_model(promoter=promoter, classifier=classifier, path=path)

  # Load dataset
  ds = pd.read_csv(filepath)
  idx = ds.index

  # Encode sequences
  X = ds[seq_col]
  X_seq = seq_train(X, promoter=promoter)
  X_seq = sm.add_constant(X_seq, has_constant='add')

  y_pred = model.predict(X_seq)

  # Store and save predictions
  if classifier:
    model_type = 'classifier'
    label = [1 if x >= 0.5 else 0 for x in y_pred]
    result = pd.DataFrame(
        {'Sequence ID': idx, '30-nt sequence': X, 'Prediction': y_pred, 'Label': label})
  else:
    model_type = 'regressor'
    result = pd.DataFrame({'Sequence ID': idx, '30-nt sequence': X, 'Prediction': y_pred})

  if save:
    result.to_csv(f'CRISPRedict_{promoter}_{model_type}_results.csv', index=None)

  return result