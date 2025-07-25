from IPython.display import Image

Image(
    "/kaggle/input/process-image/deepseek_mermaid_20250613_79aa76.png", width=500
)  # 调整宽度

# import necessary packages
import warnings

warnings.simplefilter("ignore")

import os
import gc

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
    classification_report,
    roc_auc_score,
)

import lightgbm as lgb
from lightgbm import LGBMClassifier
import logging
from sklearn.model_selection import train_test_split


def main():
    # params load
    config = load_config()
    # load the historical referrence data
    path_ref = config("path_ref")
    df = pd.read_csv(path_ref)
    df.fillna(0, inplace=True)

    buffer = []
    df.info(buf=buffer.append)
    logging.info("DataFrame Info:\n" + "\n".join(buffer))

    logging.info("DataFrame Describe:\n%s", df.describe().to_string())
    # logging.info(f"Data shape: {df.shape}")
    # logging.debug(f"\n{df.head()}")  # only works under the debug state
    # print(df.shape)
    # df.head()

    num_features = config["num_features"]
    cat_features = config["cat_features"]
    target_col = config["target_col"]
    id_col = config["id_col"]

    X = df[num_features + cat_features]
    y = dfconfig[target_col]
    user_ids = df[id_col]

    # transform type: category
    for col in cat_features:
        X[col] = X[col].astype("category")

    X_train, X_valid, y_train, y_valid, id_train, id_valid = train_test_split(
        X, y, user_ids, test_size=0.3, random_state=42
    )

    days_list = config["days_list"]
    num_features_map = config["num_features_map"]
    for i, day in enumerate(days_list, start=0):
        features = num_features_map[day] + cat_features
        globals()[f"X_train_day{day}"] = X_train[features]
        globals()[f"X_valid_day{day}"] = X_valid[features]
        globals()[f"y_train_day{day}"] = y_train.iloc[:, i]
        globals()[f"y_valid_day{day}"] = y_valid.iloc[:, i]

    payer_tag = config["payer_tag"]
    for day in days:
        print(f"\n--- Day {day} ---")

        # train datasets
        X1, X2, y1, y2 = paid_split(
            eval(f"X_train_day{day}"), eval(f"y_train_day{day}"), payer_tag
        )
        globals()[f"X_train_day{day}_1"] = X1
        globals()[f"y_train_day{day}_1"] = y1
        globals()[f"X_train_day{day}_2"] = X2
        globals()[f"y_train_day{day}_2"] = y2

        # valid datasets
        X1, X2, y1, y2 = paid_split(
            eval(f"X_valid_day{day}"), eval(f"y_valid_day{day}"), payer_tag
        )
        globals()[f"X_valid_day{day}_1"] = X1
        globals()[f"y_valid_day{day}_1"] = y1
        globals()[f"X_valid_day{day}_2"] = X2
        globals()[f"y_valid_day{day}_2"] = y2

    params_clf = config["params_clf"]
    params_reg = config["params_reg"]

    # retrain the model using valid data
    model_results = {}
    for day in days_list:
        model_results[day] = train_process(
            eval(f"X_train_day{day}_1"),
            eval(f"X_valid_day{day}_1"),
            eval(f"X_train_day{day}_2"),
            eval(f"X_valid_day{day}_2"),
            eval(f"y_train_day{day}_1"),
            eval(f"y_valid_day{day}_1"),
            eval(f"y_train_day{day}_2"),
            eval(f"y_valid_day{day}_2"),
            params_clf,
            params_reg,
            eval(f"day{day}_num_features"),
            cat_features,
        )

    # retrain the model using valid data
    model_test = {}

    for day, res in model_results.items():
        params_clf["num_iterations"] = res["model_clf"].best_iteration
        params_reg["num_iterations"] = res["model_reg"].best_iteration

        X_clf = eval(f"X_valid_day{day}_1")
        y_clf = eval(f"y_valid_day{day}_1")

        X_reg = eval(f"X_valid_day{day}_2")
        y_reg = eval(f"y_valid_day{day}_2")

        model_test[day] = train_process(
            X_clf,
            X_clf,
            X_reg,
            X_reg,
            y_clf,
            y_clf,
            y_reg,
            y_reg,
            params_clf,
            params_reg,
            eval(f"day{day}_num_features"),
            cat_features,
        )


# load the test data
path = config["path_pre"]
test_df = pd.read_csv(path_pre)
test_df.fillna(0, inplace=True)
X_test = test_df[num_features + cat_features]
y_test = test_df[target_col]
id_test = test_df[id_col]
for i, day in enumerate(days, start=0):
    features = num_features_map[day] + cat_features
    globals()[f"X_test_day{day}"] = X_test[features]
    globals()[f"y_test_day{day}"] = y_test.iloc[:, i]
for col in cat_features:
    X_test[col] = X_test[col].astype("category")

for day in days:
    X1, X2, y1, y2 = paid_split(
        eval(f"X_test_day{day}"), eval(f"y_test_day{day}"), payer_tag
    )
    globals()[f"X_test_day{day}_1"] = X1
    globals()[f"y_test_day{day}_1"] = y1
    globals()[f"X_test_day{day}_2"] = X2
    globals()[f"y_test_day{day}_2"] = y2

preds_results = {}
for day in days_list:
    preds_results[day] = predict_process(
        eval(f"X_test_day{day}_1"),
        eval(f"X_test_day{day}_2"),
        eval(f"y_test_day{day}_1"),
        eval(f"y_test_day{day}_2"),
        model_test[day]["model_clf"],
        model_test[day]["model_reg"],
    )
    # print(preds_results[day].head())
    preds_results[day].to_csv(
        f"prediction_results_DA_DAY{day}.csv", index=False, encoding="utf-8-sig"
    )

days_list_existed = config["days_list_existed"]
compare_plot(preds_results)
re_dict = {}
re_dict = evaluate_ltv(preds_results)
compara_plot(preds_results)
roas_results = show_roas_ltv(preds_results)
if __name__ == "__main__":
    main()
