"""
Prediction Module

This module handles the prediction pipeline for the LTV model, including:
- Loading trained models
- Generating predictions from classifier and regressor
- Combining results into final LTV predictions
"""

import pandas as pd
import numpy as np

# from config_loader import load_config

# config = load_config()
# payer_tag = config["payer_tag"]


def predict_process(result, model1, model2, config: dict):
    """
    Execute the full prediction process

    Args:
        x1_df (pd.DataFrame): Features for non-payer prediction
        x2_df (pd.DataFrame): Features for payer prediction
        model1 (Classifier): Trained classifier model
        model2 (Regressor): Trained regressor model
        config: Laod the config.yaml

    Returns:
        pd.DataFrame: Final predictions with LTV values
    """
    payer_tag = config["payer_tag"]
     _, _, id_test = temp_result_test["train"][day]["all"]
     x_test_nonpayer, y_test_nonpayer = temp_result_test["train"][day]["nonpayer"]
     x_test_payer, y_test_payer = temp_result_test["train"][day]["payer"]


    existing_payer_tag = [col for col in payer_tag if col in x1_df.columns]
    if not existing_payer_tag:
        raise ValueError(
            "No payer_tag columns found in X; unable to identify unpaid users."
        )
    # print(eval(f"X_test_day{day}_1").drop(columns=existing_payer_tag).head())

    temp = model1.predict(
        x1_df.drop(columns=existing_payer_tag),
        num_iteration=(
            model1.best_iteration if model1.best_iteration is not None else 100
        ),
    )

    # Deep copy
    x1_df_copy = x1_df.copy()
    x1_df_copy["pay_class_pred"] = (temp > 0.5).astype(int)

    # Filter data
    mask_payfu = x1_df_copy["pay_class_pred"] == 1
    x_df_combined = pd.concat(
        [x1_df_copy[mask_payfu].drop(columns="pay_class_pred"), x2_df], axis=0
    )
    y_combined = pd.concat([y1[mask_payfu], y2], axis=0)

    preds_log = model2.predict(
        x_df_combined,
        num_iteration=(
            model2.best_iteration if model2.best_iteration is not None else 100
        ),
    )
    preds = pd.Series(np.expm1(preds_log)).clip(lower=0).values

    x_df_combined["actual"] = y_combined.values
    x_df_combined["pred"] = preds

    # Add information for players predicted not to pay in the next step
    x1_df_copy.loc[~mask_payfu, "actual"] = y1[~mask_payfu].values
    x1_df_copy.loc[~mask_payfu, "pred"] = 0
    # print(y1[~mask_payfu].values)
    # print(x1_df_copy[~mask_payfu][['actual', 'pred']].head())

    temp = pd.concat(
        [x_df_combined, x1_df_copy[~mask_payfu].drop(columns="pay_class_pred")], axis=0
    )

    pred_results = pd.concat([id_test.rename("uid"), temp], axis=1)

    return pred_results
