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
            result_copy,
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
    # Step 6: Save metrics and plots
    # ==============================
    output_dir = create_output_dir()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    residual_dir = output_dir / "residual_plots"
    residual_dir.mkdir(exist_ok=True)

    # Save plots
    figs_com1 = compare_plot(preds_results, pre_cycles)
    for i, fig in enumerate(figs_com1):
        fig.savefig(plots_dir / f"compare_plot_cycle_{i}.png", dpi=150)
    figs_com2 = compare_plot(adjust_preds_results, pre_cycles)
    for i, fig in enumerate(figs_com2):
        fig.savefig(plots_dir / f"adjusted_compare_plot_cycle_{i}.png", dpi=150)

    figs_res1 = residual_plot(preds_results, pre_cycles)
    for i, fig in enumerate(figs_res1):
        fig.savefig(residual_dir / f"residual_plot_cycle_{i}.png", dpi=80)
    figs_res2 = residual_plot(adjust_preds_results, pre_cycles)
    for i, fig in enumerate(figs_res2):
        fig.savefig(residual_dir / f"residual_plot_adjusted_cycle_{i}.png", dpi=80)

    # Save metrics
    re_dict = evaluate_ltv(preds_results, pre_cycles)
    re_dict_adjust = evaluate_ltv(adjust_preds_results, pre_cycles)

    roas_results = show_roas_ltv(preds_results, cost, config["payer_tag"], pre_cycles)
    roas_results_adjust = show_roas_ltv(adjust_preds_results, cost, config["payer_tag"], pre_cycles)

    all_metrics = {
        "ltv": re_dict,
        "ltv_adjusted": re_dict_adjust,
        "roas": roas_results,
        "roas_adjusted": roas_results_adjust,
    }

    json_path = output_dir / "metrics_all.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    # ==============================
    # Return summary
    # ==============================
    return {
        "output_dir": str(output_dir),
        "metrics": all_metrics,
    }
