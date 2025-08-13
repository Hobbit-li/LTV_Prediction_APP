"""
Main Execution Module

Orchestrates the full model pipeline:
1. Data loading and preprocessing
2. Model training
3. Prediction generation
4. Result evaluation
"""

# from IPython.display import Image
# Image("/kaggle/input/process-image/deepseek_mermaid_20250613_79aa76.png", width=500)
# import necessary packages
from pathlib import Path
import logging
import warnings
import pandas as pd


# Local application/library specific imports
from config_loader import load_config
from data_utils import data_preprocess
from predict import predict_process
from results_show import show_roas_ltv
from train import train_process

# from utils_io import create_output_dir, save_metrics, save_model, save_predictions
from utils_io import create_output_dir, save_metrics, save_predictions
from visual import compare_plot, evaluate_ltv, residual_plot


def main():
    """
    Steps:
    1. Load configuration
    2. Preprocess data
    3. Train models
    4. Generate predictions
    5. Evaluate performance
    """
    # Configuration
    warnings.simplefilter("ignore")
    # params load
    config = load_config()
    # load the historical referrence data
    path_ref = (
        Path(__file__).parent.parent / "data" / "20250812_100327_09062_tej97.csv.gz"
    )
    df = pd.read_csv(path_ref, compression="gzip")
    df.dropna(axis=1, how="all")

    # path_ref = config["path_ref"]
    # df = pd.read_csv(path_ref)
    # df.fillna(0, inplace=True)

    buffer = []
    df.info(buf=buffer.append)
    logging.info("DataFrame Info:\n" + "\n".join(buffer))

    logging.info("DataFrame Describe:\n%s", df.describe().to_string())
    # logging.info(f"Data shape: {df.shape}")
    # logging.debug(f"\n{df.head()}")  # only works under the debug state
    # print(df.shape)
    # df.head()

    ref_month = "m5"
    cost = 10000
    # store the all splited datesets
    temp_result, pre_cycles = data_preprocess(df, config, ref_month)
    # train process
    model_results = {}

    for i in range(pre_cycles):

        result_copy = temp_result
        for split in ["train", "valid"]:
            for group in result_copy[split]:
                x, y, *rest = result_copy[split][group]
                try:
                    y = y.iloc[:, i]  # if dataframe
                except AttributeError:
                    y = [row[0] for row in y]  # if list

                result_copy[split][group] = (x, y, *rest) if rest else (x, y)

        # day_features = num_features_map[day]
        model_results[i] = train_process(
            result_copy,
            config,
        )

    # retrain the model using valid data
    model_test = {}
    model_test = model_results
    # params_clf = config["params_clf"]
    # params_reg = config["params_reg"]

    # for day, res in model_results.items():
    #     params_clf["num_iterations"] = res["model_clf"].best_iteration
    #     params_reg["num_iterations"] = res["model_reg"].best_iteration

    #     x_clf, y_clf = temp_result["valid"][day]["nonpayer"]

    #     x_reg, y_reg = temp_result["valid"][day]["payer"]

    #     model_test[day] = train_process(
    #         x_clf, x_clf, x_reg, x_reg, y_clf, y_clf, y_reg, y_reg, config
    #     )

    # load the test data
    path_pre = (
        Path(__file__).parent.parent / "data" / "20250812_100210_09037_tej97.csv.gz"
    )

    test_df = pd.read_csv(path_pre, compression="gzip")
    test_df.dropna(axis=1, how="all")

    temp_result_test = data_preprocess(test_df, config, ref_month, train_if=False)

    preds_results = {}
    adjust_preds_results = {}
    for i in range(pre_cycles):
        # use the "valid" data to predict
        result_test_copy = temp_result_test
        result_copy = temp_result
        for group in ["all", "nonpayer", "payer"]:
            x, y, *rest = result_test_copy["valid"][group]
            x1, y1, *rest1 = result_copy["valid"][group]
            try:
                y = y.iloc[:, i]  # if dataframe
                y1 = y1.iloc[:, i]
            except AttributeError:
                y = [row[0] for row in y]  # if list
                y1 = [row[0] for row in y1]

            result_test_copy["valid"][group] = (x, y, *rest) if rest else (x, y)
            result_copy["valid"][group] = (x1, y1, *rest1) if rest else (x1, y1)

        preds_results[i] = predict_process(
            result_test_copy,
            model_test[i]["model_clf"],
            model_test[i]["model_reg"],
            config,
        )

        # adjustment
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

        # print(preds_results[day].head())
        # preds_results[day].to_csv(
        #     f"prediction_results_DA_DAY{day}.csv", index=False, encoding="utf-8-sig"
        # )
    save_predictions(preds_results, create_output_dir())

    figs_com1 = compare_plot(preds_results, pre_cycles)
    figs_com2 = compare_plot(adjust_preds_results, pre_cycles)

    re_dict = {}
    re_dict_adjust = {}
    re_dict = evaluate_ltv(preds_results, pre_cycles)
    re_dict_adjust = evaluate_ltv(adjust_preds_results, pre_cycles)
    save_metrics(re_dict, create_output_dir())
    save_metrics(re_dict_adjust, create_output_dir())

    roas_results = show_roas_ltv(preds_results, cost, config["payer_tag"], pre_cycles)
    roas_results_adjust = show_roas_ltv(
        adjust_preds_results, cost, config["payer_tag"], pre_cycles
    )
    save_metrics(roas_results, create_output_dir())
    figs_res1 = residual_plot(preds_results, pre_cycles)
    figs_res2 = residual_plot(adjust_preds_results, pre_cycles)


if __name__ == "__main__":
    main()
