from IPython.display import Image
# Image("/kaggle/input/process-image/deepseek_mermaid_20250613_79aa76.png", width=500)  # adjust the width

# import necessary packages
import warnings
warnings.simplefilter('ignore')

import os
import gc

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score, classification_report, roc_auc_score

import lightgbm as lgb
from lightgbm import LGBMClassifier
import logging
from sklearn.model_selection import train_test_split

# import modules in the same path
from config_loader import load_config
from data_utils import paid_split, date_preprocess
from train import train_process
from predict import predict_process
from utils import create_output_dir, save_predictions, save_metrics, save_model
from visual import compare_plot, evaluate_ltv, residual_plot
from results_show import show_roas_ltv


def main():
  # params load
  config = load_config()
  # load the historical referrence data
  path_ref = config("path_ref")
  df = pd.read_csv(path_ref) 
  df.fillna(0, inplace=True)

 
  buffer = []  
  df.info(buf=buffer.append)
  logging.info("DataFrame Info:\n" + '\n'.join(buffer))

  logging.info("DataFrame Describe:\n%s", df.describe().to_string())
  # logging.info(f"Data shape: {df.shape}")
  # logging.debug(f"\n{df.head()}")  # only works under the debug state
  # print(df.shape)
  # df.head()

  # store the all splited datesets
  temp_result = data_preprocess(df)
  
  

 

 
  params_clf = config["params_clf"]
  params_reg = config["params_reg"]

  # retrain the model using valid data
  model_results = {}
  for day in days_list:
    X_train_nonpayer, y_train_nonpayer = temp_result['train'][day]['nonpayer']
    X_train_payer, y_train_payer       = temp_result['train'][day]['payer']
    X_valid_nonpayer, y_valid_nonpayer = temp_result['valid'][day]['nonpayer']
    X_valid_payer, y_valid_payer       = temp_result['valid'][day]['payer']
    # day_features = num_features_map[day]
    model_results[day] = train_process(
        X_train_nonpayer, X_valid_nonpayer,
        X_train_payer,    X_valid_payer,
        y_train_nonpayer, y_valid_nonpayer,
        y_train_payer,    y_valid_payer,
        params_clf, params_reg, num_features_map[day], cat_features)
  
  # retrain the model using valid data
  model_test = {}
  
  for day, res in model_results.items():
    params_clf['num_iterations'] = res["model_clf"].best_iteration
    params_reg['num_iterations'] = res["model_reg"].best_iteration
   
    X_clf, y_clf = temp_result['valid'][day]['nonpayer']
  
    X_reg, y_reg = temp_result['valid'][day]['payer']
    
    model_test[day] = train_process(
        X_clf, X_clf, X_reg, X_reg,
        y_clf, y_clf, y_reg, y_reg,
        params_clf, params_reg, num_features_map[day], cat_features
    )


  # load the test data
  path = config["path_pre"]
  test_df = pd.read_csv(path_pre) 
  test_df.fillna(0, inplace=True)
  
  temp_result_test = data_preprocess(test_df, train_data=False)
  
  preds_results = {}
  for day in days_list:
    X_test_nonpayer, y_test_nonpayer = temp_result_test['train'][day]['nonpayer']
    X_test_payer, y_test_payer       = temp_result_test['train'][day]['payer']
    
    preds_results[day] = predict_process(
          eval(X_test_nonpayer, X_test_payer, y_test_nonpayer, y_test_payer, 
          model_test[day]['model_clf'], model_test[day]['model_reg'])
      # print(preds_results[day].head())
    preds_results[day].to_csv(f"prediction_results_DA_DAY{day}.csv", index=False, encoding='utf-8-sig')
  
  days_list_existed = config["days_list_existed"]
  compare_plot(preds_results)
  re_dict = {}
  re_dict = evaluate_ltv(preds_results)
  compara_plot(preds_results)
  roas_results = show_roas_ltv(preds_results)


if __name__ == "__main__":
    main()
