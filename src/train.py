"""
Model Training Module

Contains functions for training classifier and regressor models
"""

import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    mean_squared_log_error,
    r2_score,
)

def train_clf(train_data, valid_data, config=config):
    """
    Build the Binary Classifier Model by LightGBM
    Predict whether a breakthrough payment will occur in the subsequent period
    
    Parameters:
    - train_data (tuple(pd.DataFrame, pd.Series)):
        - x_train
        - y_train
    - valid_data (tuple(pd.DataFrame, pd.Series)):
        - x_valid
        - y_valid
    - config (dict)
        - payer_tag
        - params_clf: Agrs in the classifier model
    
    return:
    - Model and model performance evaluation
    """
    x_train, y_train = train_data
    x_valid, y_valid = valid_data
    payer_tag = config["payer_tag"]
    params_clf = config["params_clf"]
    existing_payer_tag = [col for col in payer_tag if col in x_train.columns]
    if not existing_payer_tag:
        raise ValueError(
            "None of the payer_tag columns are present in x, unable to identify unpaid users."
        )

    # Feature processing
    # Keep only active dimension features
    x_tr = x_train.drop(columns=existing_payer_tag)
    x_val = x_valid.drop(columns=existing_payer_tag)
    y_tr = (y_train > 0).astype(int)
    y_val = (y_valid > 0).astype(int)
    train_set = lgb.Dataset(x_tr, label=y_tr)
    val_set = lgb.Dataset(x_val, label=y_val)

    clf = lgb.train(
        params_clf,
        train_set,
        valid_sets=[train_set, val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = list(x_tr.columns)
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["importance_gain"] = clf.feature_importance(
        importance_type="gain"
    )
    fold_importance_df = fold_importance_df.sort_values(
        by="importance", ascending=False
    )
    # fold_importance_df.to_csv(f"importance_df_clf.csv", index=None)

    # print("Example of input feature matrix for classification prediction:", x_val)
    y_pred_proba = clf.predict(x_val, num_iteration=clf.best_iteration)
    # Use probability threshold for binary classification
    y_pred = (y_pred_proba > 0.5).astype(int)

    result = {
        "classification_report": classification_report(y_val, y_pred),
        "AUC": roc_auc_score(y_val, y_pred_proba),
    }

    return clf, result


def r2_eval(preds, train_data):
    """
    Custom evaluation function to calculate R-squared (coefficient of determination) metric

    Parameters:
    - preds : array-like
        Predicted values from the model.
    - train_data : lightgbm.Dataset
        Training dataset object containing the true labels.

    Returns:
    tuple
        A tuple containing:
        - evaluation name (str): "r2"
        - r2 score (float): The calculated R-squared score
        - is_higher_better (bool): True, indicating higher scores are better
    """
    labels = train_data.get_label()
    return "r2", r2_score(labels, preds), True


def train_reg(train_data, valid_data, config=config, value_weighting=True):
    """
    Build the Regression Model by LightGBM
    Predict the value of payment will occur in the subsequent period
    
    Parameters:
    - train_data (tuple(pd.DataFrame, pd.Series)):
        - x_train
        - y_train
    - valid_data (tuple(pd.DataFrame, pd.Series)):
        - x_valid
        - y_valid
    - config (dict)
        - cat_features
        - params_reg: the agrs in the regressor model
        - percentiles: pay type
        - base_weights: weights for different pay types
        - top_num: numbers of the whale
    - value_weighting: If True, weigth the samples, default: True
        
    return:
    - Model and model performance evaluation
    """
    x_train, y_train = train_data
    x_valid, y_valid = valid_data
    params_reg = config["params_reg"]
    cat_features = config["cat_features"]
    percentiles = config["percentiles"]
    base_weights = config["base_weights"]
    top_num = config["top_num"]

    # Apply log1p transformation to target
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # Default weights are 1
    train_weights = np.ones(len(y_train))

    if value_weighting:
        bins = sorted(set([y_train.quantile(p) for p in percentiles]))

        if len(bins) < 2:
            raise ValueError(
                "Insufficient valid quantile bins, possibly all y_train values are 0 or duplicates."
            )

        labels = list(range(len(bins) - 1))
        quantile_bins = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)

        # Generate default weights of 1
        train_weights = np.ones(len(y_train))
        # Only apply weights to successfully binned samples
        mask_valid = ~quantile_bins.isna()
        quantile_ids = quantile_bins[mask_valid].astype(int)
        train_weights[mask_valid] = [base_weights[i] for i in quantile_ids]
        # Manually set the largest values (e.g., top 10 samples) to 1000
        top_k_idx = np.argsort(y_train.values)[-top_num:]  # Top 10 largest samples
        train_weights[top_k_idx] = base_weights[-1]  # Force the value to 1000

    # print(pd.value_counts(train_weights, sort=False))

    # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    counts = pd.value_counts(train_weights, sort=False)
    logging.info("Train weight counts:\n%s", counts.to_string())

    features = num_features + cat_features

    # Wrap dataset
    trn_data = lgb.Dataset(
        x_train,
        label=y_train_log,
        categorical_feature=cat_features,
        weight=train_weights,
    )
    val_data = lgb.Dataset(
        x_valid, label=y_valid_log, categorical_feature=cat_features, reference=trn_data
    )
    reg = lgb.train(
        params_reg,
        trn_data,
        valid_sets=[trn_data, val_data],
        feval=r2_eval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=500),
        ],
    )  # Equivalent to verbose_eval=100

   
    # Predict log values
    y_preds_log = reg.predict(x_valid, num_iteration=reg.best_iteration)
    # Restore log and correct negative values
    y_preds = pd.Series(np.expm1(y_preds_log)).clip(lower=0).values

    # Model evaluation metrics -- Mean Squared Log Error and R2 Score
    result = {
        "MSLE": mean_squared_log_error(y_valid, y_preds),
        "R2": r2_score(y_valid, y_preds),
    }

    return reg, result


def train_process(
    result_df,
    config=config,
):
    """
    The binary classification model determines future payment behavior
    The regression model predicts future LTV
    
    Parameters:
    - resultd_df: the original and splited datsset
    - config:
        - params_clf:
        - params_reg
    
    """
    x_train_1, y_train_1 = result_df["train"]["nonpayer"]
    x_train_2, y_train_2 = result_df["train"]["payer"]
    x_valid_1, y_valid_1 = result_df["valid"]["nonpayer"]
    x_valid_2, y_valid_2 = result_df["valid"]["payer"]
   
    payer_tag = config["payer_tag"]
    # Check if the validation set is empty
    # The purpose is to train only, without validation
    only_train = any(df is None for df in [x_valid_1, x_valid_2, y_valid_1, y_valid_2])
    if only_train:
        # Construct empty DataFrames with the same structure
        x_valid_1 = pd.DataFrame(columns=x_train_1.columns)
        x_valid_2 = pd.DataFrame(columns=x_train_2.columns)
        y_valid_1 = pd.Series(dtype="float64")
        y_valid_2 = pd.Series(dtype="float64")
        # print("Train only, no validation")

    # Train the classification model on the dataset of players who have not paid during the feature period
    clf_valid, result_valid_clf= train_clf(
        result_df["train"]["nonpayer"], result_df["valid"]["nonpayer"]
    )

    # Predict on dataset 1
    existing_payer_tag = [col for col in payer_tag if col in x_train_1.columns]
    if not existing_payer_tag:
        raise ValueError(
            "None of the payer_tag columns are present in x, unable to identify unpaid users."
        )
    temp = clf_valid.predict(x_train_1.drop(columns=existing_payer_tag))

    # Create deep copy
    x_train_1_copy = x_train_1.copy()
    x_train_1_copy["pay_class_pred"] = (temp > 0.5).astype(int)

    # eval(f"x_train_day{day}_1")['pay_class_pred'] = (temp > 0.5).astype(int)
    # print(eval(f"x_train_day{day}_1").drop(columns=existing_payer_tag).head())

    # Create deep copy
    x_valid_1_copy = x_valid_1.copy()
    if not x_valid_1.empty:
        temp = clf_valid.predict(x_valid_1.drop(columns=existing_payer_tag))
        x_valid_1_copy["pay_class_pred"] = (temp > 0.5).astype(int)
    else:
        x_valid_1_copy["pay_class_pred"] = pd.Series(dtype="int")
    # eval(f"x_valid_day{day}_1")['pay_class_pred'] = (temp > 0.5).astype(int)

    # Filter data
    mask_payfu_1 = x_train_1_copy["pay_class_pred"] == 1
    mask_payfu_2 = x_valid_1_copy["pay_class_pred"] == 1

    # Concatenate data
    x_combined_train = pd.concat(
        [x_train_1_copy[mask_payfu_1].drop(columns="pay_class_pred"), x_train_2], axis=0
    )
    y_combined_train = pd.concat([y_train_1[mask_payfu_1], y_train_2], axis=0)

    x_combined_valid = pd.concat(
        [x_valid_1_copy[mask_payfu_2].drop(columns="pay_class_pred"), x_valid_2], axis=0
    )
    y_combined_valid = pd.concat([y_valid_1[mask_payfu_2], y_valid_2], axis=0)

    reg_valid, result_valid_reg, importance_reg = train_reg(
        (x_combined_train, y_combined_train), (x_combined_valid, y_combined_valid),
    )
    model_result = {
        "model_clf": clf_valid,
        "result_clf": result_valid_clf,
        "model_reg": reg_valid,
        "result_reg": result_valid_reg,
    }

    # If only training the model without validation
    # Only return the models
    if only_train:
        model_result.update(
            {
                "model_clf": clf_valid,
                "model_reg": reg_valid,
            }
        )
    return model_result
