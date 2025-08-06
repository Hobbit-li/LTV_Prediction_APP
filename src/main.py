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
    temp_result, pre_cycles = data_preprocess(df, config)
    # train process
    model_results = {}
    result_copy = temp_result.copy()
    for i in range(pre_cycles):
        
        for split in ["train", "valid"]:
            for group in result_copy[split]:
                x, y, *rest = result_copy[split][group]
                try:
                    y = y.iloc[:, i]  # 如果是 DataFrame
                except AttributeError:
                    y = [row[0] for row in y]  # 如果是列表

                result_copy[split][group] = (x, y, *rest) if rest else (x, y)

        # day_features = num_features_map[day]
        model_results[i] = train_process(
           result_copy,
            config,
        )

    # retrain the model using valid data
    model_test = {}
    model_test = model_results
    params_clf = config["params_clf"]
    params_reg = config["params_reg"]

    # for day, res in model_results.items():
    #     params_clf["num_iterations"] = res["model_clf"].best_iteration
    #     params_reg["num_iterations"] = res["model_reg"].best_iteration

    #     x_clf, y_clf = temp_result["valid"][day]["nonpayer"]

    #     x_reg, y_reg = temp_result["valid"][day]["payer"]

    #     model_test[day] = train_process(
    #         x_clf, x_clf, x_reg, x_reg, y_clf, y_clf, y_reg, y_reg, config
    #     )

    # load the test data
    path_pre = config["path_pre"]
    test_df = pd.read_csv(path_pre)
    test_df.fillna(0, inplace=True)

    temp_result_test = data_preprocess(test_df, config, train_data=False)
    result_test_copy = temp_result_test.copy()

    preds_results = {}
    for i in range(pre_cycles):
        
        for group in result_copy['train']:
                x, y, *rest = result_test_copy['train'][group]
                try:
                    y = y.iloc[:, i]  # 如果是 DataFrame
                except AttributeError:
                    y = [row[0] for row in y]  # 如果是列表

                result_test_copy['train'][group] = (x, y, *rest) if rest else (x, y)
            

        preds_results[day] = predict_process(
            result_test_copy,
            model_test[day]["model_clf"],
            model_test[day]["model_reg"],
            config,
        )
        # print(preds_results[day].head())
        # preds_results[day].to_csv(
        #     f"prediction_results_DA_DAY{day}.csv", index=False, encoding="utf-8-sig"
        # )
    save_predictions(preds_results, create_output_dir())

    compare_plot(preds_results, config)
    re_dict = {}
    re_dict = evaluate_ltv(preds_results, config)
    save_metrics(re_dict, create_output_dir())
    compare_plot(preds_results, config)
    roas_results = show_roas_ltv(preds_results, config)
    save_metrics(roas_results, create_output_dir())
    residual_plot(preds_results, config)


if __name__ == "__main__":
    main()
