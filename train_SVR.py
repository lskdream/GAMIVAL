# -*- coding: utf-8 -*-
'''
This script shows how to train and validate regression model and the optimized parameters will be saved automatically

python train_SVR.py \
  --model_name GAME \
  --dataset_name LIVE-Meta-Gaming \
  --feature_file feat_files/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_bicubic_feats.mat \
  --best_parameter best_pamtr/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_pamtr \
  --use_parallel

'''

import pandas
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
  parser.add_argument('--log_file', type=str,
                      default='logs/LIVE-Meta-Mobile-Cloud-Gaming_GAMIVAL_SVR.log',
                      help='Log files.')
  parser.add_argument('--log_short', action='store_true',
                      help='Whether log short')
  parser.add_argument('--use_parallel', action='store_true',
                      help='Use parallel for iterations.')
  parser.add_argument('--max_thread_count', type=int, default=4,
                      help='Number of threads.')
  args = parser.parse_args()
  return args

def evaluate_bvqa_kfold(X, y, log_short):
  if not log_short:
    t_start = time.time()
  
  # grid search CV on the training set
  print(f'{X.shape[1]}-dim features, using SVR')
  # grid search CV on the training set
  param_grid = {'C': np.logspace(1, 10, 10, base=2),
                'gamma': np.logspace(-10, -6, 5, base=2)}
  grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=8, n_jobs=4, verbose=2)

  scaler = preprocessing.MinMaxScaler().fit(X)
  X = scaler.transform(X)
  grid.fit(X, y)
  best_params = grid.best_params_
  return best_params

def evaluate_bvqa_kfold_linearSVR(X, y, log_short):
  if not log_short:
    t_start = time.time()

  # grid search CV on the training set
  print(f'{X.shape[1]}-dim features, using LinearSVR')
  # grid search on liblinear 
  param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
  grid = GridSearchCV(LinearSVR(random_state=1, max_iter=100), param_grid, n_jobs=4, cv=8, verbose=2)
  scaler = preprocessing.MinMaxScaler().fit(X)
  X = scaler.transform(X)
  # grid search
  grid.fit(X, y)
  best_params = grid.best_params_
  return best_params
    
def main(args):
  csv_file = os.path.join('mos_files', args.dataset_name+'_metadata.csv')
  df = pandas.read_csv(csv_file)
  y = df['MOS'].to_numpy()

  y = np.array(list(y), dtype=np.float)
  X_mat = scipy.io.loadmat(args.feature_file)
  X = np.asarray(X_mat['feats_mat'], dtype=np.float)

  ## preprocessing
  X[np.isinf(X)] = np.nan
  imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
  X = imp.transform(X)

  t_overall_start = time.time()

  if X.shape[1] <= 4000:
    best_params_SVR = evaluate_bvqa_kfold(X, y, args.log_short)
  best_params_linearSVR = evaluate_bvqa_kfold_linearSVR(X, y, args.log_short)

  print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))
  # save figures

  dir_path = os.path.dirname(args.best_parameter)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  scipy.io.savemat(args.best_parameter+'_SVR.mat', 
      mdict={'best_parameters': np.asarray(best_params_SVR,dtype=object)})
  scipy.io.savemat(args.best_parameter+'_linearSVR.mat', 
      mdict={'best_parameters': np.asarray(best_params_linearSVR,dtype=object)})

if __name__ == '__main__':
  args = arg_parser()
  log_dir = os.path.dirname(args.log_file)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  sys.stdout = Logger(args.log_file)
  print(args)
  main(args)
