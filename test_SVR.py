# -*- coding: utf-8 -*-
'''
This script shows how to predict the quality score by pretrained SVR/linearSVR

python test_SVR.py \
  --model_name GAME \
  --dataset_name LIVE-Meta-Gaming \
  --feature_file feat_files/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_bicubic_feats.mat \
  --best_parameter best_pamtr/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_pamtr \
  --use_parallel

'''
__test__ = False  # Prevent pytest from collecting this script as a test

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
                      help='BVQA model name.')
  parser.add_argument('--dataset_name', type=str, default='LIVE-Meta-Gaming',
                      help='Trained dataset.') 
  parser.add_argument('--feature_file', type=str,
                      default='feat_files/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_bicubic_feats.mat',
                      help='Pre-computed feature matrix.')
  parser.add_argument('--best_parameter', type=str,
                      default='best_pamtr/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_pamtr',
                      help='Output best parameters')
  parser.add_argument('--predicted_score', type=str,
                      default='predicted_score/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_predicted_score',
                      help='Output predicted scores')
  parser.add_argument('--log_file', type=str,
                      default='logs/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_predict.log',
                      help='Log files.')
  parser.add_argument('--log_short', action='store_true',
                      help='Whether log short')
  parser.add_argument('--use_parallel', action='store_true',
                      help='Use parallel for iterations.')
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

def evaluate_bvqa_kfold(X, y, best_params, log_short):
  if not log_short:
    t_start = time.time()
  
  # init model
  regressor = SVR(C=best_params['C'], gamma=best_params['gamma'])
  # re-train the model using the best alpha
  regressor.fit(X, y)
  # predictions
  y_pred = regressor.predict(X)
  # compute metrics
  metrics, y_pred_logistic = compute_metrics(y_pred, y)
  # print values
  if not log_short:
    print('{} secs lapsed..'.format(time.time() - t_start))
  return metrics, y_pred_logistic

def evaluate_bvqa_kfold_linearSVR(X, y, best_params, log_short):
  if not log_short:
    t_start = time.time()

  # init model
  regressor = LinearSVR(C=best_params['C'], epsilon=best_params['epsilon'])
  # re-train the model using the best alpha
  regressor.fit(X, y)
  # predictions
  y_pred = regressor.predict(X)
  # compute metrics
  metrics, y_pred_logistic = compute_metrics(y_pred, y)
  # print values
  if not log_short:
    print('{} secs lapsed..'.format(time.time() - t_start))
  return metrics, y_pred_logistic
    
def main(args):
  csv_file = os.path.join('mos_files', args.dataset_name+'_metadata.csv')
  df = pandas.read_csv(csv_file)
  y = df['MOS'].to_numpy()

  y = np.array(list(y), dtype=np.float)
  X_mat = scipy.io.loadmat(args.feature_file)
  X = np.asarray(X_mat['feats_mat'], dtype=np.float)

  best_params_mat = scipy.io.loadmat(args.best_parameter+'_SVR.mat')
  best_params_SVR = {'C': np.asarray(best_params_mat['best_parameters'][0,0][0][0][0], dtype=np.float),
                     'gamma': np.asarray(best_params_mat['best_parameters'][0,0][0][0][1], dtype=np.float)}
  best_params_mat = scipy.io.loadmat(args.best_parameter+'_linearSVR.mat')
  best_params_linearSVR = {'C': np.asarray(best_params_mat['best_parameters'][0,0][0][0][0], dtype=np.float),
                     'epsilon': np.asarray(best_params_mat['best_parameters'][0,0][0][0][1], dtype=np.float)}
  
  ## preprocessing
  X[np.isinf(X)] = np.nan
  imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
  X = imp.transform(X)

  t_overall_start = time.time()

  if X.shape[1] <= 4000:
    metrics_SVR, y_pred_logistic_SVR = evaluate_bvqa_kfold(X, y, best_params_SVR, args.log_short)
  metrics_linearSVR, y_pred_logistic_linearSVR = evaluate_bvqa_kfold_linearSVR(X, y, best_params_linearSVR, args.log_short)

  print(metrics_SVR)
  print(metrics_linearSVR)
  print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))
    
  dir_path = os.path.dirname(args.predicted_score)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  scipy.io.savemat(args.predicted_score+'_SVR.mat', 
      mdict={'predicted_score': np.asarray(y_pred_logistic_SVR,dtype=np.float)})
  scipy.io.savemat(args.predicted_score+'_linearSVR.mat', 
      mdict={'predicted_score': np.asarray(y_pred_logistic_linearSVR,dtype=np.float)})

if __name__ == '__main__':
  args = arg_parser()
  log_dir = os.path.dirname(args.log_file)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  sys.stdout = Logger(args.log_file)
  print(args)
  main(args)
