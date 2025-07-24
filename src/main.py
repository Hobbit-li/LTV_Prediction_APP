from IPython.display import Image
Image("/kaggle/input/process-image/deepseek_mermaid_20250613_79aa76.png", width=500)  # 调整宽度

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
