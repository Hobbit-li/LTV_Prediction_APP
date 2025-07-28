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
import pandas as pd
import logging
# Local application/library specific imports
from config_loader import load_config
from data_utils import data_preprocess
from predict import predict_process
from results_show import show_roas_ltv
from train import train_process

# from utils_io import create_output_dir, save_metrics, save_model, save_predictions
from utils_io import create_output_dir, save_predictions
from visual import compare_plot, evaluate_ltv, residual_plot


def main():
    """
    Execute end-to-end model workflow

    Steps:
    1. Load configuration
    2. Preprocess data
    3. Train models
    4. Generate predictions
    5. Evaluate performance
    6. Save outputs
    """
    # Configuration
    warnings.simplefilter("ignore")
    pd.set_option("display.max_columns", None)
    # params load
    config = load_config()
    # load the historical referrence data
    path_ref = config["path_ref"]
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

    # store the all splited datesets
    temp_result = data_preprocess(df)
    days_list = config["days_list"]
    # train process
    model_results = {}
    for day in days_list:
        x_train_nonpayer, y_train_nonpayer = temp_result["train"][day]["nonpayer"]
        x_train_payer, y_train_payer = temp_result["train"][day]["payer"]
        x_valid_nonpayer, y_valid_nonpayer = temp_result["valid"][day]["nonpayer"]
        x_valid_payer, y_valid_payer = temp_result["valid"][day]["payer"]
        # day_features = num_features_map[day]
        model_results[day] = train_process(
            x_train_nonpayer,
            x_valid_nonpayer,
            x_train_payer,
            x_valid_payer,
            y_train_nonpayer,
            y_valid_nonpayer,
            y_train_payer,
            y_valid_payer,
            config,
        )

    # retrain the model using valid data
    model_test = {}
    params_clf = config["params_clf"]
    params_reg = config["params_reg"]

    for day, res in model_results.items():
        params_clf["num_iterations"] = res["model_clf"].best_iteration
        params_reg["num_iterations"] = res["model_reg"].best_iteration

        x_clf, y_clf = temp_result["valid"][day]["nonpayer"]

        x_reg, y_reg = temp_result["valid"][day]["payer"]

        model_test[day] = train_process(
            x_clf, x_clf, x_reg, x_reg, y_clf, y_clf, y_reg, y_reg, config
        )

    # load the test data
    path_pre = config["path_pre"]
    test_df = pd.read_csv(path_pre)
    test_df.fillna(0, inplace=True)

    temp_result_test = data_preprocess(test_df, train_data=False)

    preds_results = {}
    for day in days_list:
        x_test_nonpayer, y_test_nonpayer = temp_result_test["train"][day]["nonpayer"]
        x_test_payer, y_test_payer = temp_result_test["train"][day]["payer"]

        preds_results[day] = predict_process(
            x_test_nonpayer,
            x_test_payer,
            y_test_nonpayer,
            y_test_payer,
            model_test[day]["model_clf"],
            model_test[day]["model_reg"],
        )
        # print(preds_results[day].head())
        # preds_results[day].to_csv(
        #     f"prediction_results_DA_DAY{day}.csv", index=False, encoding="utf-8-sig"
        # )
    save_predictions(preds_results, create_output_dir())

    compare_plot(preds_results, config)
    re_dict = {}
    re_dict = evaluate_ltv(preds_results, config)
    compare_plot(preds_results, config)
    roas_results = show_roas_ltv(preds_results, config)
    residual_plot(preds_results, config)


if __name__ == "__main__":
    main()
