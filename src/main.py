"""
Main Execution Module

Orchestrates the full model pipeline:
1. Data loading and preprocessing
2. Model training
3. Prediction generation
4. Result evaluation
"""

import io
from pathlib import Path
import logging
import sys
import warnings
import pandas as pd

# Local application/library specific imports
from config_loader import load_config
from data_utils import data_preprocess
from predict import predict_process
from results_show import show_roas_ltv
from train import train_process
from utils_io import create_output_dir, save_metrics, save_predictions
from visual import compare_plot, evaluate_ltv, residual_plot

# ==============================
# Logging Setup
# ==============================
logs_path = Path(__file__).parent.parent / "logs"
logs_path.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # INFO 或 DEBUG 可根据需求调整
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_path / "main_debug.log", mode="w"),
    ],
)


def main():
    warnings.simplefilter("ignore")

    # ==============================
    # Step 1: Load configuration
    # ==============================
    logging.info("Step 1: Load configuration")
    config = load_config()

    # ==============================
    # Step 2: Load reference data
    # ==============================
    logging.info("Step 2: Load reference data")
    path_ref = (
        Path(__file__).parent.parent / "data" / "20250812_100327_09062_tej97.csv.gz"
    )
    df = pd.read_csv(path_ref, compression="gzip")
    df.dropna(axis=1, how="all", inplace=True)

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    logging.info("DataFrame Info:\n%s", info_str)
    logging.info("DataFrame Describe:\n%s", df.describe().to_string())
    logging.debug("DataFrame head:\n%s", df.head())

    ref_month = "m5"
    cost = 1234992

    # ==============================
    # Step 3: Data preprocessing
    # ==============================
    logging.info("Step 3: Data preprocessing")
    temp_result, pre_cycles = data_preprocess(df, config, ref_month)
    logging.info(f"Preprocessing done, pre_cycles={pre_cycles}")

    # 打印 temp_result 结构和部分内容
    logging.debug("temp_result structure and sample content:")
    logging.debug(f"pre_cycles:{pre_cycles}")
    for split in temp_result:
        logging.debug(f"Split: {split}")
        for group in temp_result[split]:
            x, y, *rest = temp_result[split][group]
            y_type = type(y)
            logging.debug(f"  Group: {group}, y type: {y_type}")
            # 打印 y 的前 5 个元素，避免日志过长
            if isinstance(y, (pd.DataFrame, pd.Series)):
                logging.debug(f"    y head:\n{y.head()}")
            elif isinstance(y, list):
                logging.debug(f"    y sample: {y[:5]}")
            else:
                logging.debug(f"    y content (type {y_type}): {str(y)[:100]}")

    # ==============================
    # Step 4: Training models
    # ==============================
    logging.info("Step 4: Training models")
    model_results = {}
    for i in range(pre_cycles):
        logging.debug(f"Training cycle {i}")
        result_copy = temp_result
        for split in ["train", "valid"]:
            for group in result_copy[split]:
                x, y, *rest = result_copy[split][group]
                try:
                    y = y.iloc[:, i]  # if dataframe
                except AttributeError:
                    y = [row[0] for row in y]  # if list
                result_copy[split][group] = (x, y, *rest) if rest else (x, y)
        model_results[i] = train_process(result_copy, config)

    # ==============================
    # Step 5: Prepare test data
    # ==============================
    logging.info("Step 5: Preparing test data")
    path_pre = (
        Path(__file__).parent.parent / "data" / "20250812_100210_09037_tej97.csv.gz"
    )
    test_df = pd.read_csv(path_pre, compression="gzip")
    test_df.dropna(axis=1, how="all", inplace=True)
    temp_result_test, _ = data_preprocess(test_df, config, ref_month, train_if=False)

    # ==============================
    # Step 6: Generate predictions
    # ==============================
    logging.info("Step 6: Generating predictions")
    preds_results = {}
    adjust_preds_results = {}
    model_test = model_results

    for i in range(pre_cycles):
        logging.debug(f"Prediction cycle {i}")
        result_test_copy = temp_result_test
        result_copy = temp_result
        for group in ["all", "nonpayer", "payer"]:
            x, y, *rest = result_test_copy["valid"][group]
            x1, y1, *rest1 = result_copy["valid"][group]
            try:
                y = y.iloc[:, i]
                y1 = y1.iloc[:, i]
            except AttributeError:
                y = [row[0] for row in y]
                y1 = [row[0] for row in y1]
            result_test_copy["valid"][group] = (x, y, *rest) if rest else (x, y)
            result_copy["valid"][group] = (x1, y1, *rest1) if rest else (x1, y1)

        preds_results[i] = predict_process(
            result_test_copy,
            model_test[i]["model_clf"],
            model_test[i]["model_reg"],
            config,
        )

        preds_train = predict_process(
            result_copy,
            model_test[i]["model_clf"],
            model_test[i]["model_reg"],
            config,
        )
        adjustment = (
            preds_train["pred"].values.sum() - preds_train["actual"].values.sum()
        ) / len(preds_train)
        adjust_preds_results[i] = preds_results[i].copy()
        adjust_preds_results[i]["pred"] = adjust_preds_results[i]["pred"] - adjustment

    # ==============================
    # Step 7: Save predictions
    # ==============================
    logging.info("Step 7: Saving predictions")
    output_dir = create_output_dir()
    output_dir.mkdir(exist_ok=True, parents=True)

    for i, df_pred in preds_results.items():
        csv_path = output_dir / f"predictions_cycle_{i}.csv"
        df_pred.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logging.info(f"Saved predictions CSV: {csv_path}")

    excel_path = output_dir / "predictions_all.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        for i, df_pred in preds_results.items():
            df_pred.to_excel(writer, sheet_name=f"cycle_{i}", index=False)
    logging.info(f"Saved all predictions Excel: {excel_path}")

    # ==============================
    # Step 8: Save plots
    # ==============================
    logging.info("Step 8: Generating and saving plots")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    figs_com1 = compare_plot(preds_results, pre_cycles)
    for i, fig in enumerate(figs_com1):
        png_path = plots_dir / f"compare_plot_cycle_{i}.png"
        fig.savefig(png_path, dpi=150)
        logging.info(f"Saved compare plot PNG: {png_path}")

    figs_com2 = compare_plot(adjust_preds_results, pre_cycles)
    for i, fig in enumerate(figs_com2):
        png_path = plots_dir / f"adjusted_compare_plot_cycle_{i}.png"
        fig.savefig(png_path, dpi=150)
        logging.info(f"Saved adjusted compare plot PNG: {png_path}")

    # ==============================
    # Step 9: Save LTV / ROAS metrics
    # ==============================
    logging.info("Step 9: Saving LTV and ROAS metrics")

    # Evaluate LTV
    re_dict = evaluate_ltv(preds_results, pre_cycles)
    re_dict_adjust = evaluate_ltv(adjust_preds_results, pre_cycles)

    # 保存 LTV metrics
    for name, metrics_dict in zip(["ltv", "ltv_adjusted"], [re_dict, re_dict_adjust]):
        for key, df_metric in metrics_dict.items():
            csv_path = output_dir / f"{name}_{key}.csv"
            df_metric.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logging.info(f"Saved {name} metric CSV: {csv_path}")

    # Show ROAS LTV
    roas_results = show_roas_ltv(preds_results, cost, config["payer_tag"], pre_cycles)
    roas_results_adjust = show_roas_ltv(
        adjust_preds_results, cost, config["payer_tag"], pre_cycles
    )

    # 保存 ROAS metrics
    for name, df_metric in zip(
        ["roas", "roas_adjusted"], [roas_results, roas_results_adjust]
    ):
        csv_path = output_dir / f"{name}.csv"
        df_metric.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logging.info(f"Saved {name} CSV: {csv_path}")

    # ==============================
    # Step 10: Save residual plots
    # ==============================
    logging.info("Step 10: Saving residual plots")
    residual_dir = output_dir / "residual_plots"
    residual_dir.mkdir(exist_ok=True)

    figs_res1 = residual_plot(preds_results, pre_cycles)
    figs_res2 = residual_plot(adjust_preds_results, pre_cycles)

    for i, fig in enumerate(figs_res1):
        png_path = residual_dir / f"residual_plot_cycle_{i}.png"
        fig.savefig(png_path, dpi=150)
        logging.info(f"Saved residual plot PNG: {png_path}")

    for i, fig in enumerate(figs_res2):
        png_path = residual_dir / f"residual_plot_adjusted_cycle_{i}.png"
        fig.savefig(png_path, dpi=150)
        logging.info(f"Saved adjusted residual plot PNG: {png_path}")


if __name__ == "__main__":
    main()
