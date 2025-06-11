# -*- coding: utf-8 -*-
'''
This script performs an 80-20 holdout evaluation of a regression model (SVR or LinearSVR)
for predicting MOS (Mean Opinion Scores) from extracted BVQA features.
'''

# Import necessary packages
import argparse
import os, sys, time, math, warnings
import numpy as np
import scipy.io
import scipy.stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

try:
    import pandas
except ImportError:
    pandas = None

warnings.filterwarnings("ignore")  # Suppress warnings

# Logger class to redirect stdout to a log file
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# Argument parser

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='GAME')
    parser.add_argument('--dataset_name', type=str, default='LIVE-Meta-Gaming')
    parser.add_argument('--feature_file', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--predicted_score', type=str)
    parser.add_argument('--best_parameter', type=str)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--log_short', action='store_true')
    parser.add_argument('--use_parallel', action='store_true')
    parser.add_argument('--num_iterations', type=int, default=6)
    parser.add_argument('--max_thread_count', type=int, default=4)
    return parser.parse_args()

# 4-parameter logistic function for curve fitting
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(-(X - bayta3) / abs(bayta4))
    yhat = bayta2 + (bayta1 - bayta2) / logisticPart
    return yhat

# Compute quality prediction metrics
def compute_metrics(y_pred, y):
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, KRCC, PLCC, RMSE], y_pred_logistic

# Pretty-print evaluation results
def formatted_print(snapshot, params, duration):
    print('======================================================')
    print('params: ', params)
    print('SRCC_train: ', snapshot[0])
    print('KRCC_train: ', snapshot[1])
    print('PLCC_train: ', snapshot[2])
    print('RMSE_train: ', snapshot[3])
    print('======================================================')
    print('SRCC_test: ', snapshot[4])
    print('KRCC_test: ', snapshot[5])
    print('PLCC_test: ', snapshot[6])
    print('RMSE_test: ', snapshot[7])
    print('======================================================')
    print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

# Aggregate metrics across iterations
def final_avg(snapshot):
    def formatted(args, pos):
        mean = np.mean([x[pos] for x in snapshot])
        median = np.median([x[pos] for x in snapshot])
        stdev = np.std([x[pos] for x in snapshot])
        print('{}: {} (median: {}) (std: {})'.format(args, mean, median, stdev))

    print('======================================================')
    print('Average training results among all repeated 80-20 holdouts:')
    formatted("SRCC Train", 0)
    formatted("KRCC Train", 1)
    formatted("PLCC Train", 2)
    formatted("RMSE Train", 3)
    print('======================================================')
    print('Average testing results among all repeated 80-20 holdouts:')
    formatted("SRCC Test", 4)
    formatted("KRCC Test", 5)
    formatted("PLCC Test", 6)
    formatted("RMSE Test", 7)
    print('\n\n')

# Train and evaluate a BVQA regression model
# Uses either SVR or LinearSVR based on feature dimensions
# Includes grid search for hyperparameter tuning
def evaluate_bvqa_kfold(X_train, X_test, y_train, y_test, log_short):
    if not log_short:
        t_start = time.time()

    if X_train.shape[1] <= 4000:
        param_grid = {'C': np.logspace(1, 10, 10, base=2),
                      'gamma': np.logspace(-10, -6, 5, base=2)}
        grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=8, n_jobs=4, verbose=2)
    else:
        param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                      'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
        grid = GridSearchCV(LinearSVR(random_state=1, max_iter=100), param_grid, n_jobs=4, cv=8, verbose=2)

    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    regressor = SVR(**best_params) if X_train.shape[1] <= 4000 else LinearSVR(**best_params)
    regressor.fit(X_train, y_train)

    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)
    metrics_train, _ = compute_metrics(y_train_pred, y_train)
    metrics_test, y_test_pred_logistic = compute_metrics(y_test_pred, y_test)

    if not log_short:
        t_end = time.time()
        formatted_print(metrics_train + metrics_test, best_params, (t_end - t_start))

    return best_params, metrics_train, metrics_test, y_test_pred_logistic

# Main script entry point
def main(args):
    # Load MOS scores and features
    csv_file = os.path.join('mos_files', args.dataset_name + '_metadata.csv')
    df = pandas.read_csv(csv_file)
    y = df['MOS'].astype(float).to_numpy()
    X = scipy.io.loadmat(args.feature_file)['feats_mat'].astype(float)

    # Impute missing values (NaN or Inf)
    X[np.isinf(X)] = np.nan
    X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)

    all_iterations = []
    best_parameters = []
    t_overall_start = time.time()

    # Repeated 80-20 split evaluation (for content-aware datasets)
    if args.dataset_name in ['GamingVideoSET', 'KUGVD', 'CGVDS', 'LIVE-Meta-Gaming']:
        content = df['Content'].to_numpy()
        for i in range(args.num_iterations):
            with open(args.dataset_name + '_idx.npy', 'rb') as f:
                train_content = np.load(f, allow_pickle=True)[i]
                test_content = np.load(f, allow_pickle=True)[i]

            X_train, y_train, X_test, y_test = [], [], [], []
            for c in range(len(content)):
                if content[c] in train_content:
                    X_train.append(X[c])
                    y_train.append(y[c])
                if content[c] in test_content:
                    X_test.append(X[c])
                    y_test.append(y[c])

            X_train, y_train = np.asarray(X_train), np.asarray(y_train)
            X_test, y_test = np.asarray(X_test), np.asarray(y_test)

            best_params, metrics_train, metrics_test, y_test_pred = evaluate_bvqa_kfold(X_train, X_test, y_train, y_test, args.log_short)
            all_iterations.append(metrics_train + metrics_test)
            best_parameters.append(best_params)

    # Print and save results
    final_avg(all_iterations)
    print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    np.save(args.out_file + ".npy", np.asarray(all_iterations))
    scipy.io.savemat(args.out_file + '.mat', {'all_iterations': np.asarray(all_iterations)})

    os.makedirs(os.path.dirname(args.best_parameter), exist_ok=True)
    scipy.io.savemat(args.best_parameter + '_' + str(args.num_iterations) + 'iter.mat',
                    {'best_parameters': np.asarray(best_parameters, dtype=object)})

if __name__ == '__main__':
    args = arg_parser()
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    sys.stdout = Logger(args.log_file)
    print(args)
    main(args)
