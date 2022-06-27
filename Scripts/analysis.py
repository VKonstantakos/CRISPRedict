import math
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sns.set()


def Spearman(X, Y):
    '''
    Calculates Spearman correlation coefficient between X and Y.

    :parameters X, Y: parameters for correlation analysis.
    :return: Spearman correlation coefficient.
    '''

    coeff, _ = stats.spearmanr(X, Y)
    return coeff


def spearman_ds(ds, actual_col, pred_col, p_value=False, nan_policy='omit'):
    '''
    Calculates Spearman correlation coefficient between actual and predicted efficiencies.
    By default, omits NaN values and outputs only the coefficient without the corresponding p-value.

    :parameter ds: dataset to analyse.
    :parameter actual_col: # position of column with actual efficiencies e.g. 2nd column -> 2.
    :parameter pred_col: # position of column with each model's predictions.
    :parameter p_value: If True, calculate the two-sided p-value for a hypothesis test whose null hypothesis is that two sets of data are uncorrelated.
    :return: Spearman correlation coefficient (or a tuple of Spearman correlation coefficient and corresponding p-value).
    '''

    Spearman, pval = stats.spearmanr(
        ds.iloc[:, actual_col - 1], ds.iloc[:, pred_col - 1], nan_policy=nan_policy)
    if p_value == False:
        return Spearman
    else:
        return Spearman, pval


def ndcg_at_k(ds, k, actual_col, pred_col, bins=False, reverse=False, multiple=False):
    '''
    Calculates nDCG@k score using logarithmic discount given a dataset with actual and predicted efficiencies.
    The relevance value of each gRNA is its efficacy score.

    :parameter ds: dataset to analyse.
    :parameter k: highest value to calculate nDCG i.e. only consider the highest k scores in the ranking.
    :parameter actual_col: # position of column with actual efficiencies e.g. 2nd column -> 2.
    :parameter pred_col: # position of column with each model's predictions.
    :parameter bins (default=False): If True, group actual efficiencies into 5 equal width bins to have a discrete relevance value.
                                     By default, use actual efficiency as the relevance value.
    :parameter reverse (default=False): If True, calculate nDCG for reverse ordering to be used as a baseline.
    :parameter multiple (default=False): If True, calculate and store nDCG together with samples' indices for plotting use.
    :return: nDCG@k score (or a list of scores and indices if parameter multiple = True).
    '''

    # Discrete relevance value

    if bins == True:
        quantile_list = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
        bins = ds.iloc[:, actual_col-1].quantile(quantile_list)
        labels = [0, 1, 2, 3, 4]
        ds['binned'] = pd.cut(ds.iloc[:, actual_col-1],
                              bins, labels=labels, include_lowest=True)

    # Reverse ordering (worst case)
        if reverse == True:
            ds_sort = ds.sort_values(by=ds.columns[actual_col-1])
            y_pred = ds_sort['binned'].values

    # Ordering based on each model's predictions
        else:
            ds_sort = ds.sort_values(
                by=ds.columns[pred_col-1], ascending=False)
            y_pred = ds_sort['binned'].values

    # Actual relevance value

    else:

        # Reverse ordering (worst case)
        if reverse == True:
            ds_sort = ds.sort_values(by=ds.columns[actual_col-1])
            y_pred = ds_sort.iloc[:, actual_col-1].values

    # Ordering based on each model's predictions
        else:
            ds_sort = ds.sort_values(
                by=ds.columns[pred_col-1], ascending=False)
            y_pred = ds_sort.iloc[:, actual_col-1].values

    # Ideal ordering
    Y_test = sorted(y_pred, reverse=True)

    # Calculate nDCG@k

    thr = []
    score = []

    a = b = 0
    for i in range(0, k):
        a += y_pred[i]/math.log(i+2, 2)
        b += Y_test[i]/math.log(i+2, 2)
        thr.append(i)
        score.append(a/b)

    if multiple == True:
        return score, thr
    else:
        return a/b


def r_precision(ds, actual_col, pred_col, label_col=None, labeled=False, percentile=80, threshold=None):
    '''
    Calculates R-Precision between actual and predicted efficiencies.

    :parameter ds: dataset to analyse.
    :parameter actual_col: # position of column with actual efficiencies e.g. 2nd column -> 2.
    :parameter pred_col: # position of column with each model's predictions.
    :parameter label_col: # position of column with labels.
    :parameter labeled: option to use labels for relevance.
    :parameter percentile: percentile threshold of actual efficiency to consider as relevant. Must be between 0 and 100 inclusive.
    :parameter threshold: absolute threshold of actual efficiency to consider as relevant. Default is None.
    :return: R-Precision and number of relevant results R.
    '''

    # Retrieve results based on predictions
    ds_sort = ds.sort_values(by=ds.columns[pred_col-1], ascending=False)

    # Use labels to determine relevant results, if available
    if labeled == True:

        try:
            r = np.asarray(ds_sort.iloc[:, label_col-1])
        except ValueError:
            print('No column label was selected. Provide a column with the necessary labels.')

    # If labels are not available, use a threshold to determine relevant results
    else:

        # Determine percentile-based threshold
        if percentile:
            threshold = np.percentile(ds.iloc[:, actual_col-1], percentile)
        elif threshold:
            threshold = threshold
        else:
            raise ValueError('Input a threshold to determine relevant efficiencies.')

        # Split into relevant and non-relevant based on threshold
        r = np.asarray(ds_sort.iloc[:, actual_col-1]) >= threshold

    # Calculate R-Precision
    z = np.sum(r)

    if not z:
        return 0.

    return np.mean(r[:z]), z


def auc_ds(ds, label_col, pred_col):
    '''
    Calculates Area under the ROC Curve for each classifier.

    :parameter ds: dataset to analyse.
    :parameter label_col: # position of column with labels e.g. 3rd column -> 3.
    :parameter pred_col: # position of column with each model's predictions.
    :return: AUC.
    '''

    # Calculate AUC using the classifier's predictions
    auc = roc_auc_score(ds.iloc[:, label_col-1], ds.iloc[:, pred_col-1])

    return auc


def roc_curve_ds(ds, label_col, pred_col):
    '''
    Calculates and plots Receiver Operating Characteristic (ROC) Curve for each classifier.

    :parameter ds: dataset to analyse.
    :parameter label_col: # position of column with labels e.g. 3rd column -> 3.
    :parameter pred_col: # position of column with each model's predictions.
    :return: ROC curve plot.
    '''

    # Calculate ROC curve and AUC for each classifier
    fpr, tpr, _ = roc_curve(ds.iloc[:, label_col-1], ds.iloc[:, pred_col-1])
    auc = roc_auc_score(ds.iloc[:, label_col-1], ds.iloc[:, pred_col-1])

    # Plot the results
    plt.plot(fpr, tpr, label=f'{ds.columns[pred_col-1]}, AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.legend(prop={'size': 13}, loc='lower right')


def tost_test(model_name_1='CRISPRedict', model_name_2='DeepHF'):
  '''
  Performs Two One-Sided Test (TOST) to determine equivalence between the performances of two models.
  By default, it compares the performance of CRISPRedict and DeepHF.

  :parameters model_name_1, model_name_2: the names of the compared models (e.g., CRISPRedict, DeepHF, TSAM).
  :return: dictionary with the evaluation metrics and the corresponding p-values.
  '''

  # Load results
  spear = pd.read_csv(
      '../Data/4. Results/1. Regression/Spearman Results.csv', index_col=0)
  ndcg = pd.read_csv(
      '../Data/4. Results/1. Regression/nDCG Results.csv', index_col=0)
  rprec = pd.read_csv(
      '../Data/4. Results/1. Regression/R-Precision Results.csv', index_col=0)
  rprec_lab = pd.read_csv(
      '../Data/4. Results/2. Classification/R-Precision (Labeled-NaN) Results.csv', index_col=0)
  auc_res = pd.read_csv(
      '../Data/4. Results/2. Classification/AUC (Labeled-NaN) Results.csv', index_col=0)

  # Drop NaN
  auc_res = auc_res.dropna(axis=1)
  rprec_lab = rprec_lab.dropna(axis=1)

  # Define datasets
  results = [spear, ndcg, rprec, rprec_lab, auc_res]
  names = ['Spearman correlation', 'nDCG@(n/5)', 'R-Precision', 'R-Precision (Labeled)', 'AUC']
  pvals = []

  for name, data in zip(names, results):
      print(f'\n\n{name} between {model_name_1} and {model_name_2}')
      print('-'*(len(name)+len(model_name_1)+len(model_name_2)+15))
      if pg.normality(data[model_name_1])['normal'][0] == True and pg.normality(data[model_name_2])['normal'][0] == True:
          print('Normality assumption is valid.\n')

          pg.print_table(
              pg.tost(data[model_name_1], data[model_name_2], bound=0.075, paired=True))
          pvals.append(pg.tost(
              data[model_name_1], data[model_name_2], bound=0.075, paired=True)['pval'][0])

      else:
          print('Normality assumption is violated.')

  return dict(zip(names, pvals))
