"""
Main Execution Logic (GUI-adapted)

Provides a function `run_pipeline` that:
1. Loads input data from uploaded file
2. Runs preprocessing, training, prediction
3. Computes and saves metrics/plots
4. Returns summary results (dict)
"""

import io
from pathlib import Path
import logging
import warnings
import pandas as pd
import copy
import json

# Local application imports
from config_loader import load_config
from data_utils import data_preprocess
from predict import predict_process
from results_show import show_roas_ltv
from train import train_process
from utils_io import create_output_dir
from visual import compare_plot, evaluate_ltv, residual_plot


def run_pipeline(path_ref: str, path_pre: str, ref_month: str, cost: float):
    """
    Run the full pipeline for given file and parameters.
    Returns a dict of results (ROAS, LTV, etc.).
    """

    warnings.simplefilter("ignore")

    # ==============================
    # Step 1: Load configuration
    # ==============================
    config = load_config()

    # ==============================
    # Step 2: Load reference data
    # ==============================
    df = pd.read_csv(path_ref) if path_ref.endswith(".csv") else pd.read_excel(path_ref)
    df.dropna(axis=1, how="all", inplace=True)

    # ==============================
    # Step 3: Data preprocessing
    # ==============================
    temp_result, pre_cycles = data_preprocess(df, config, ref_month)

    # ==============================
    # Step 4: Training models
    # ==============================
    model_results = {}
    for i in range(pre_cycles):
        result_copy = copy.deepcopy(temp_result)
        for split in ["train", "valid"]:
            for group in result_copy[split]:
                x, y, *rest = result_copy[split][group]
                try:
                    y = y.iloc[:, i].fillna(0)  # if dataframe
                except AttributeError:
                    y = [row[0] for row in y]  # if list
                result_copy[split][group] = (x, y, *rest) if rest else (x, y)
        model_results[i] = train_process(result_copy, config)

    # ==============================
    # Step 5: Predictions
    # ==============================
    preds_results = {}
    adjust_preds_results = {}
    test_df = pd.read_csv(path_pre) if path_pre.endswith(".csv") else pd.read_excel(path_pre)
    test_df.dropna(axis=1, how="all", inplace=True)

    temp_result_test, _ = data_preprocess(test_df, config, ref_month, train_if=False)
    for i in range(pre_cycles):
        result_test_copy = copy.deepcopy(temp_result_test)
        result_copy = copy.deepcopy(temp_result)
        for group in ["all", "nonpayer", "payer"]:
            x, y, *rest = result_test_copy["valid"][group]
            x1, y1, *rest1 = result_copy["valid"][group]
            try:
                if hasattr(y, "iloc") and i < y.shape[1]:
                    y = y.iloc[:, i].fillna(0)
                else:
                    y = pd.Series([0] * len(y), index=y.index)
            except Exception:
                y = pd.Series([0] * len(y), index=y.index)
            try:
                if hasattr(y1, "iloc") and i < y1.shape[1]:
                    y1 = y1.iloc[:, i].fillna(0)
                else:
                    y1 = pd.Series([0] * len(y1), index=y1.index)
            except Exception:
                y1 = pd.Series([0] * len(y1), index=y1.index)

            result_test_copy["valid"][group] = (x, y, *rest) if rest else (x, y)
            result_copy["valid"][group] = (x1, y1, *rest1) if rest else (x1, y1)

        preds_results[i] = predict_process(
            result_test_copy,
            model_results[i]["model_clf"],
            model_results[i]["model_reg"],
            config,
        )

        preds_train = predict_process(
            result_copy,
            model_results[i]["model_clf"],
            model_results[i]["model_reg"],
            config,
        )
        adjustment = (
            preds_train["pred"].values.sum() - preds_train["actual"].values.sum()
        ) / len(preds_train)
        adjust_preds_results[i] = preds_results[i].copy()
        adjust_preds_results[i]["pred"] = adjust_preds_results[i]["pred"] - adjustment

    # ==============================
    # Step 6: show metrics and plots
    # ==============================

    figs_com1 = compare_plot(preds_results, pre_cycles)
    figs_com2 = compare_plot(adjust_preds_results, pre_cycles)
    figs_res1 = residual_plot(preds_results, pre_cycles)
    figs_res2 = residual_plot(adjust_preds_results, pre_cycles)

    # Save metrics
    re_dict = evaluate_ltv(preds_results, pre_cycles)
    re_dict_adjust = evaluate_ltv(adjust_preds_results, pre_cycles)

    roas_results = show_roas_ltv(preds_results, cost, config["payer_tag"], pre_cycles)
    roas_results_adjust = show_roas_ltv(adjust_preds_results, cost, config["payer_tag"], pre_cycles)
    roas_df1 = pd.DataFrame([
        {
            "Subsequent Months": i+1,
            "ROAS_actual": v["ROAS_actual"],
            "ROAS_pred": v["ROAS_pred"],
            "LTV_actual": v["LTV_actual"],
            "LTV_pred": v["LTV_pred"],
        } for i, v in roas_results.items()
    ])
    roas_df2 = pd.DataFrame([
        {
            "Subsequent Months": i+1,
            "ROAS_actual": v["ROAS_actual"],
            "ROAS_pred_adjusted": v["ROAS_pred"],
            "LTV_actual": v["LTV_actual"],
            "LTV_pred_adjusted": v["LTV_pred"],
        } for i, v in roas_results_adjust.items()
    ])
     merged_df = pd.merge(
        roas_df1,
        roas_df2,
        on="Subsequent Months",
        suffixes=("_df1", "_df2")
    )

    # 判断是否一致
    roas_equal = merged_df["ROAS_actual_df1"].equals(merged_df["ROAS_actual_df2"])
    ltv_equal = merged_df["LTV_actual_df1"].equals(merged_df["LTV_actual_df2"])

    if not roas_equal:
        warnings.warn("⚠️ ROAS_actual 在两个 DataFrame 中不一致！")
    if not ltv_equal:
        warnings.warn("⚠️ LTV_actual 在两个 DataFrame 中不一致！")

    # 整理列顺序
    predictions_df = merged_df[
        [
            "Subsequent Months",
            "ROAS_actual_df1",
            "ROAS_pred",
            "ROAS_pred_adjusted",
            "LTV_actual_df1",
            "LTV_pred",
            "LTV_pred_adjusted",
        ]
    ].rename(
        columns={
            "ROAS_actual_df1": "ROAS_actual",
            "LTV_actual_df1": "LTV_actual",
        }
    )

    all_metrics = {
        "ltv": re_dict,
        "ltv_adjusted": re_dict_adjust,
    }

    # Encode plots
    plots_base64 = {
        "compare_plots": [fig_to_base64(f) for f in figs_com1],
        "compare_plots_adjusted": [fig_to_base64(f) for f in figs_com2],
        "residual_plots": [fig_to_base64(f) for f in figs_res1],
        "residual_plots_adjusted": [fig_to_base64(f) for f in figs_res2],
    }


    # ==============================
    # Return summary
    # ==============================
    return {
        "model_evaluate": all_metrics,
        "ROAS_LTV predictions": predictions_df,
        "plots_show": plots_base64,
    }
