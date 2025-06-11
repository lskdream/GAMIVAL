# -*- coding: utf-8 -*-
'''
This script shows how to apply 80-20 holdout train and validate regression model to predict
MOS from the features

python evaluate_bvqa_features_regression.py \
  --model_name GAME \
  --dataset_name LIVE-Meta-Gaming \
  --feature_file feat_files/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_bicubic_feats.mat \
  --out_file result/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_corr \
  --best_parameter best_pamtr/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_pamtr \
  --use_parallel

'''

try:
  import pandas
except ImportError:  # Allow importing without pandas for testing
  pandas = None
import scipy.io
import numpy as np
import argparse
import time
import math
import os, sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import scipy.stats
from concurrent import futures
import functools
import warnings
import random
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
# ----------------------- Set System logger ------------- #
class Logger:
  def __init__(self, log_file):
    self.terminal = sys.stdout
    self.log = open(log_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    pass


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='GAME',
                      help='Evaluated BVQA model name.')
  parser.add_argument('--dataset_name', type=str, default='LIVE-Meta-Gaming',
                      help='Evaluation dataset.') 
  parser.add_argument('--feature_file', type=str,
                      default='feat_files/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_bicubic_feats.mat',
                      help='Pre-computed feature matrix.')
  parser.add_argument('--out_file', type=str,
                      default='result/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_corr',
                      help='Output correlation results')
  parser.add_argument('--predicted_score', type=str,
                      default='predicted_score/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_predicted_score.mat',
                      help='Output predicted scores')
  parser.add_argument('--best_parameter', type=str,
                      default='best_pamtr/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR_pamtr',
                      help='Output best parameters')
  parser.add_argument('--log_file', type=str,
                      default='logs/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR.log',
                      help='Log files.')
  parser.add_argument('--log_short', action='store_true',
                      help='Whether log short')
  parser.add_argument('--use_parallel', action='store_true',
                      help='Use parallel for iterations.')
  parser.add_argument('--num_iterations', type=int, default=6,
                      help='Number of iterations of train-test splits')
  parser.add_argument('--max_thread_count', type=int, default=4,
                      help='Number of threads.')
  args = parser.parse_args()
  return args

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

def compute_metrics(y_pred, y):
  '''
  compute metrics btw predictions & labels
  '''
  # compute SRCC & KRCC
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  # logistic regression btw y_pred & y
  beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
  popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)
  
  # compute  PLCC RMSE
  PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
  return [SRCC, KRCC, PLCC, RMSE], y_pred_logistic

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

def final_avg(snapshot):
  def formatted(args, pos):
    mean = np.mean(list(map(lambda x: x[pos], snapshot)))
    median = np.median(list(map(lambda x: x[pos], snapshot)))
    stdev = np.std(list(map(lambda x: x[pos], snapshot)))
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

def evaluate_bvqa_kfold(X_train, X_test, y_train, y_test, log_short):
  if not log_short:
    #print('{} th repeated holdout test'.format(i))
    t_start = time.time()
  
  # grid search CV on the training set
  if X_train.shape[1] <= 4000:
    print(f'{X_train.shape[1]}-dim features, using SVR')
    # grid search CV on the training set
    param_grid = {'C': np.logspace(1, 10, 10, base=2),
                  'gamma': np.logspace(-10, -6, 5, base=2)}
    #grid = RandomizedSearchCV(SVR(), param_grid, cv=5, n_jobs=4)
    grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=8, n_jobs=4, verbose=2)
  else:
    print(f'{X_train.shape[1]}-dim features, using LinearSVR')
    # grid search on liblinear 
    param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                  'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
    grid = GridSearchCV(LinearSVR(random_state=1, max_iter=100), param_grid, n_jobs=4, cv=8, verbose=2)
  scaler = preprocessing.MinMaxScaler().fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  grid.fit(X_train, y_train)
  best_params = grid.best_params_
  # init model
  if X_train.shape[1] <= 4000:
    regressor = SVR(C=best_params['C'], gamma=best_params['gamma'])
  else:
    regressor = LinearSVR(C=best_params['C'], epsilon=best_params['epsilon'])
  # re-train the model using the best alpha
  regressor.fit(X_train, y_train)
  # predictions
  y_train_pred = regressor.predict(X_train)
  y_test_pred = regressor.predict(X_test)
  # compute metrics
  metrics_train, y_train_pred_logistic = compute_metrics(y_train_pred, y_train)
  metrics_test, y_test_pred_logistic = compute_metrics(y_test_pred, y_test)
  # print values
  if not log_short:
    t_end = time.time()
    formatted_print(metrics_train + metrics_test, best_params, (t_end - t_start))
  return best_params, metrics_train, metrics_test, y_test_pred_logistic

def evaluate_bvqa_kfold_linearSVR(X_train, X_test, y_train, y_test, log_short):
  if not log_short:
    #print('{} th repeated holdout test'.format(i))
    t_start = time.time()

  # grid search CV on the training set
  print(f'{X_train.shape[1]}-dim features, using LinearSVR')
  # grid search on liblinear 
  param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
  grid = GridSearchCV(LinearSVR(random_state=1, max_iter=100), param_grid, n_jobs=4, cv=8, verbose=2)
  scaler = preprocessing.MinMaxScaler().fit(X_train)
  X_train = scaler.transform(X_train)
  # grid search
  grid.fit(X_train, y_train)
  best_params = grid.best_params_
  # init model
  regressor = LinearSVR(C=best_params['C'], epsilon=best_params['epsilon'])
  # re-train the model using the best alpha
  regressor.fit(X_train, y_train)
  # predictions
  y_train_pred = regressor.predict(X_train)
  y_test_pred = regressor.predict(X_test)
  # compute metrics
  metrics_train, y_train_pred_logistic = compute_metrics(y_train_pred, y_train)
  metrics_test, y_test_pred_logistic = compute_metrics(y_test_pred, y_test)
  # print values
  if not log_short:
    t_end = time.time()
    formatted_print(metrics_train + metrics_test, best_params, (t_end - t_start))
  return best_params, metrics_train, metrics_test, y_test_pred_logistic
    
def main(args):
  csv_file = os.path.join('mos_files', args.dataset_name+'_metadata.csv')
  df = pandas.read_csv(csv_file)
  y = df['MOS'].to_numpy()
  if args.dataset_name == 'LIVE-Meta-Gaming' or args.dataset_name == 'KUGVD' or args.dataset_name == 'GamingVideoSET' or args.dataset_name == 'CGVDS':
      content = df['Content'].to_numpy()

  y = np.array(list(y), dtype=np.float)
  X_mat = scipy.io.loadmat(args.feature_file)
  X = np.asarray(X_mat['feats_mat'], dtype=np.float)
  #X = np.concatenate((X_[:,0:102],X_[:,136:306],X_[:,544:-1]),axis=1)

  ## preprocessing
  X[np.isinf(X)] = np.nan
  imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
  X = imp.transform(X)

  all_iterations = []
  y_pred = np.array([])
  best_parameters = []
  t_overall_start = time.time()

  # 100 times random train-test splits
  if args.dataset_name == 'LIVE-YT-Gaming' or args.dataset_name == 'YT-UGC-Gaming':
    for i in range(0, args.num_iterations):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=math.ceil(8.8*i))
      best_params, metrics_train, metrics_test, y_test_pred = evaluate_bvqa_kfold(X_train, X_test, y_train, y_test, args.log_short)
      if np.isnan(metrics_test[0]) or np.isnan(metrics_test[1]) or np.isnan(metrics_test[2]) :
         best_params, metrics_train, metrics_test, y_test_pred = evaluate_bvqa_kfold_linearSVR(X_train, X_test, y_train, y_test, args.log_short)
      all_iterations.append(metrics_train + metrics_test)

  elif args.dataset_name == 'GamingVideoSET' or args.dataset_name == 'KUGVD' or args.dataset_name == 'CGVDS' or args.dataset_name == 'LIVE-Meta-Gaming':    
    for i in range(0, args.num_iterations):
      f = open(args.dataset_name + '_idx.npy', 'rb')
      train_content = np.load(f,allow_pickle=True)[i]
      test_content = np.load(f,allow_pickle=True)[i]
      f.close()
        
      X_train = []
      X_test = []
      y_train = []
      y_test = []
        
      for c in range(content.shape[0]):
        if content[c] in train_content:
            X_train.append(X[c,:])
            y_train.append(y[c])

        if content[c] in test_content:
            X_test.append(X[c,:])
            y_test.append(y[c])

      X_train = np.asarray(X_train)
      X_test = np.asarray(X_test)
      y_train = np.asarray(y_train)
      y_test = np.asarray(y_test)

      best_params, metrics_train, metrics_test, y_test_pred = evaluate_bvqa_kfold(X_train, X_test, y_train, y_test, args.log_short)
      if np.isnan(metrics_test[0]) or np.isnan(metrics_test[1]) or np.isnan(metrics_test[2]) :
         best_params, metrics_train, metrics_test, y_test_pred = evaluate_bvqa_kfold_linearSVR(X_train, X_test, y_train, y_test, args.log_short)
      all_iterations.append(metrics_train + metrics_test)
      best_parameters.append(best_params)
      print(best_parameters)
      #y_pred = np.append(y_pred,y_test_pred)

  # formatted print overall iterations
  final_avg(all_iterations)
  print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))
  
  dir_path = os.path.dirname(args.out_file)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  all_iterations = np.asarray(all_iterations,dtype=np.float)
  np.save(args.out_file+".npy",all_iterations)
  scipy.io.savemat(args.out_file+'.mat', 
      mdict={'all_iterations': all_iterations})

  '''
  dir_path = os.path.dirname(args.predicted_score)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  scipy.io.savemat(args.predicted_score, 
      mdict={'predicted_score': np.asarray(y_test_pred,dtype=np.float)})
  '''

  dir_path = os.path.dirname(args.best_parameter)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  scipy.io.savemat(args.best_parameter+'_'+str(args.num_iterations)+'iter.mat',
      mdict={'best_parameters': np.asarray(best_parameters,dtype=object)})

if __name__ == '__main__':
  args = arg_parser()
  log_dir = os.path.dirname(args.log_file)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  sys.stdout = Logger(args.log_file)
  print(args)
  main(args)
