import json
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from genetic_selection import GeneticSelectionCV
from sklearn.feature_selection import SelectFromModel, RFECV

sns.set()


def feature_importance(model, X, Y, threshold=None, save=False):
    '''
    Select features based on coefficients/importance weights.

    :parameter model: model from which the transformer is built.
    :parameter X: feature set.
    :parameter Y: target variable.
    :parameter threshold: threshold to use for feature selection.
    :parameter save: option to save the selected features.
    :return: transformed feature set.
    '''

    # Initialize and apply transformer
    feat_sel = SelectFromModel(model, threshold=threshold)

    tic = time()

    feat_sel.fit(X, Y)

    toc = time()

    # Transform X feature set
    X = X.iloc[:, feat_sel.get_support()]

    # Save features
    if save == True:
        features = X.columns.to_list()

        with open("features_importance.txt", "w") as f:
            json.dump(features, f)

    print(f"Done in {(toc - tic)/60:0.2f} minutes")
    print(f"New feature shape: {X.shape}")
    return X


def rfecv_selection(model, X, Y, cv=10, scoring=None, figure=False, save=False):
    '''
    Feature ranking with recursive feature elimination and
    cross-validated selection of the best number of features.

    :parameter model: model from which the transformer is built.
    :parameter X: feature set.
    :parameter Y: target variable.
    :parameter cv: determines the cross-validation splitting strategy.
    :parameter scoring: scoring function for evaluation.
    :parameter figure: option to show/save the RFECV graph.
    :parameter save: option to save the selected features.
    :return: transformed feature set.
    '''

    # Initialize and apply RFECV
    rfecv = RFECV(model, step=1, cv=cv, scoring=scoring)

    tic = time()

    rfecv.fit(X.values, Y)

    toc = time()

    # Transform X feature set
    X = X.iloc[:, rfecv.support_]

    print(f"Done in {(toc - tic)/60:0.2f} minutes")
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"New feature shape: {X.shape}")

    # Save features
    if save == True:
        features = X.columns.to_list()

        with open("features_rfecv.txt", "w") as f:
            json.dump(features, f)

    # Plot RFECV search
    if figure == True:

        # Plot number of features VS. cross-validation scores
        plt.figure(figsize=(16,9))

        plt.xlabel("Number of features selected", fontsize=15)
        plt.ylabel("Cross validation score", fontsize=15)
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])

        plt.savefig('RFECV.png')
        plt.show()

    return X


def ga_selection(model, X, Y, cv=3, scoring=None, max_features=None, n_population=4000,
                n_generations=500, crossover_independent_proba=0.5, n_gen_no_change=15,
                verbose=1, n_jobs=1, save=False):
    '''
    Select features using Genetic Algorithm search.

    :parameter model: model from which the transformer is built.
    :parameter X: feature set.
    :parameter Y: target variable:
    :parameter cv: determines the cross-validation splitting strategy.
    :parameter scoring: scoring function for evaluation.
    :parameter max_features: maximum number of features selected.
    :parameter n_population: population number for the genetic algorithm.
    :parameter n_generations: number of generations for the genetic algorithm.
    :parameter crossover_independent_proba: independent probability for each attribute to be exchanged.
    :parameter n_gen_no_change: number of generations with no change to terminate searching.
    :parameter verbose: controls verbosity of output.
    :parameter n_jobs: number of cores to run in parallel.
    :parameter save: option to save the selected features.
    :return: transformed feature set.
    '''

    # Initialize and apply transformer
    feat_sel = GeneticSelectionCV(model, cv=cv, scoring=scoring, max_features=max_features,
                                  n_population=n_population, n_generations=n_generations,
                                  crossover_independent_proba=crossover_independent_proba,
                                  n_gen_no_change=n_gen_no_change, verbose=verbose, n_jobs=n_jobs)

    tic = time()

    feat_sel.fit(X, Y)

    toc = time()

    # Transform X feature set
    X = X.iloc[:, feat_sel.support_]

    # Save features
    if save == True:
        features = X.columns.to_list()

        with open(f"features_ga_{max_features}.txt", "w") as f:
            json.dump(features, f)

    print(f"Done in {(toc - tic)/60:0.2f} minutes")
    print(f"New feature shape: {X.shape}")
    return X
