# -*- coding: utf-8 -*-
'''
This script evaluates BVQA features using SVR regression under different noise levels.
It performs grid search for hyperparameter tuning and evaluates model performance over 100 repeated train/test splits.

Command-line interface (CLI) version.
'''

import argparse
import os
import warnings
import numpy as np
import scipy.io
import pandas as pd
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import scipy.stats
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Logistic function for non-linear regression curve fitting
def logistic_func(X, b1, b2, b3, b4):
    logisticPart = 1 + np.exp(-(X - b3) / abs(b4))
    yhat = b2 + (b1 - b2) / logisticPart
    return yhat

# Compute standard performance metrics
def compute_metrics(y_pred, y):
    y_pred[np.isnan(y_pred)] = 0
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]
    try:
        beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
        popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
        y_pred_logistic = logistic_func(y_pred, *popt)
    except:
        y_pred_logistic = y_pred
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return SRCC, KRCC, PLCC, RMSE

# Split training and testing based on content ID

def traintest_split(train_content, test_content, content, scores, features):
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(content)):
        if content[i] in train_content:
            X_train.append(features[i])
            y_train.append(scores[i])
        if content[i] in test_content:
            X_test.append(features[i])
            y_test.append(scores[i])
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)

# Evaluate model performance for one random split
def train_test_run(r, content, scores, features, index_file):
    print(f"Running split {r}")
    with open(index_file, 'rb') as f:
        train_content = np.load(f, allow_pickle=True)[r]
        test_content = np.load(f, allow_pickle=True)[r]

    X_train, y_train, X_test, y_test = traintest_split(train_content, test_content, content, scores, features)
    best_C, best_gamma = grid_search(X_train, y_train)

    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVR(gamma=best_gamma, C=best_C)
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    train_metrics = compute_metrics(preds_train, y_train)
    test_metrics = compute_metrics(preds_test, y_test)
    return train_metrics + test_metrics

# Perform grid search for best hyperparameters
def grid_search(X, y):
    gamma_list = np.logspace(-8, 2, 10)
    C_list = np.logspace(1, 10, 10, base=2)
    best_score = -np.inf
    best_C = C_list[0]
    best_gamma = gamma_list[0]

    for gamma in gamma_list:
        for C in C_list:
            model = SVR(gamma=gamma, C=C)
            scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
            X_scaled = scaler.transform(X)
            scores_cv = []
            for cv_index in range(5):
                X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=cv_index)
                model.fit(X_tr, y_tr)
                scores_cv.append(model.score(X_val, y_val))
            mean_score = np.mean(scores_cv)
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_gamma = gamma
    return best_C, best_gamma

# Main CLI execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='LIVE-Meta-Gaming')
    parser.add_argument('--algo_name', type=str, default='GAMIVAL')
    parser.add_argument('--noise_levels', nargs='+', default=['001', '005'])
    parser.add_argument('--index_file', type=str, default='content_idx.npy')
    parser.add_argument('--feature_dir', type=str, default='features')
    parser.add_argument('--output_dir', type=str, default='result_cli')
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for noise in args.noise_levels:
        print(f"Evaluating noise level: {noise}")
        csv_file = f"{args.dataset_name}_metadata.csv"
        mat_file = os.path.join(args.feature_dir, f"{args.dataset_name}_{args.algo_name}_feats_w{noise}.mat")

        df = pd.read_csv(csv_file)
        scores = df['MOS'].to_numpy()
        content = df['Content'].to_numpy()

        X_mat = scipy.io.loadmat(mat_file)
        features = np.asarray(X_mat['feats_mat'], dtype='float64')
        features[np.isinf(features)] = np.nan
        features = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(features)

        metrics_list = Parallel(n_jobs=args.n_jobs)(
            delayed(train_test_run)(i, content, scores, features, args.index_file) for i in range(args.num_runs)
        )

        metrics_array = np.asarray(metrics_list)
        output_path = os.path.join(args.output_dir, f"metrics_{args.algo_name}_w{noise}.npy")
        np.save(output_path, metrics_array)

        print(f"Mean Metrics: {np.mean(metrics_array, axis=0)}")
        print(f"Median Metrics: {np.median(metrics_array, axis=0)}")
        print(f"STD Metrics: {np.std(metrics_array, axis=0)}")

if __name__ == '__main__':
    main()
