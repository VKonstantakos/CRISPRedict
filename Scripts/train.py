import numpy as np
import pandas as pd
import statsmodels.api as sm
from features import *


def get_training_data(promoter='u6', classifier=False, add_constant=True, path='../Data/2. Training datasets'):
  '''
  Loads training dataset for the corresponding model.

  :parameter promoter: gRNA transcription method to compute relevant features ('u6', 't7').
  :parameter classifier: option to transform dataset for classification. Default is False.
  :parameter add_constant: option to add constant variable (e.g., for statsmodels library).
  :parameter path: file path containing the training datasets.
  :return: feature set X and target variable Y.
  '''

  if promoter == 'u6':
    # Load dataset
    ds = pd.read_excel(
        f'{path}/DeepSpCas9.xlsx')

    # Define X
    X = ds['Target context sequence (4+20+3+3)']

    # Encode sequences
    X = seq_train(X, promoter=promoter)

    if classifier == False:
      # Define success and failure from reads
      Y_success = ds['Indel read count\n(Day 2.9)']
      Y_failure = ds['Total read count\n(Day 2.9)'] - ds['Indel read count\n(Day 2.9)']

      Y = pd.DataFrame({'Success': Y_success, 'Failure': Y_failure})

    elif classifier == True:
      # Define Y
      Y = ds['Indel frequencies (%)']/100

      # Transform dataset for classification
      perc_20 = np.percentile(Y, 20)
      perc_80 = np.percentile(Y, 80)

      Y_inef = Y[Y < perc_20]
      Y_eff = Y[Y > perc_80]
      Y_inef.loc[:] = 0
      Y_eff.loc[:] = 1

      Y = pd.concat([Y_inef, Y_eff], axis=0).sort_index(ascending=True)
      X = X.loc[Y.index, :]

    else:
      raise ValueError(
          "Select a valid model type (classifier is True or False).")

  elif promoter == 't7':
    # Load dataset
    ds = pd.read_csv(
        f'{path}/Moreno-Mateos.csv')

    # Define X, Y
    X = ds['30-nt sequence']
    Y = ds['modFreq']

    # Encode sequences
    X = seq_train(X, promoter=promoter)

    if classifier == False:
      pass

    elif classifier == True:
      # Transform dataset for classification
      perc_20 = np.percentile(Y, 20)
      perc_80 = np.percentile(Y, 80)

      Y_inef = Y[Y < perc_20]
      Y_eff = Y[Y > perc_80]
      Y_inef.loc[:] = 0
      Y_eff.loc[:] = 1

      Y = pd.concat([Y_inef, Y_eff], axis=0).sort_index(ascending=True)
      X = X.loc[Y.index, :]

    else:
      raise ValueError(
          "Select a valid model type (classifier is True or False).")

  else:
      raise ValueError(
          "No promoter was selected. Select a valid promoter ('u6' or 't7') to calculate the relevant features.")

  # Add constant variable
  if add_constant == True:
    X = sm.add_constant(X)

  return X, Y


def train_model(promoter=None, classifier=False, path='../Data/2. Training datasets', save=True):
  '''
  Creates the appropriate model for the chosen parameters.

  :parameter promoter: gRNA transcription method to create the suitable model ('u6', 't7').
  :parameter classifier: option to train a classification model. Default is False.
  :parameter path: file path containing the training datasets.
  :parameter save: option to save the created model. Default is True.
  :return: trained model.
  '''

  if classifier == True:
    model_type = 'classifier'
    X, Y = get_training_data(promoter=promoter, classifier=classifier, path=path)
    model = sm.Logit(Y, X).fit()

  elif classifier == False:
    model_type = 'regressor'
    if promoter == 'u6':
      X, Y = get_training_data(promoter=promoter, classifier=classifier, path=path)
      glm_binom = sm.GLM(Y, X, family=sm.families.Binomial())
      model = glm_binom.fit()

    elif promoter == 't7':
      X, Y = get_training_data(promoter=promoter, classifier=classifier, path=path)
      model = sm.OLS(Y, X).fit()

    else:
      raise ValueError(
          "No promoter was selected. Select a valid promoter ('u6' or 't7') to continue.")

  else:
    raise ValueError(
        "Select a valid model type (classifier is True or False).")

  if save:
    model.save(f'CRISPRedict_{promoter}_{model_type}.pickle')

  return model
