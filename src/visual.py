"""
Visualization Module

Provides visualization tools for model evaluation including:
- Performance comparison plots
- LTV evaluation charts
- Residual analysis plots
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
)


def compare_plot(preds_results, cycles=pre_cycles):
    """
    Objective: show the actual and the predicted LTV increasingly
    days_list_existed: have existed days for valid
    preds_results: the predicted results by running the trained model
    """
    for i in range(cycles):
        plt.figure(figsize=(20, 6))
        result_df_sorted = (
            preds_results[i].sort_values(by="actual").reset_index(drop=True)
        )
        plt.plot(result_df_sorted["actual"], label="Actual", color="blue", linewidth=2)
        plt.plot(
            result_df_sorted["pred"],
            label="Predicted",
            color="orange",
            linewidth=2,
            linestyle="--",
        )

        plt.title(f"Model Prediction vs Actual--Month Cycle--{i+1}")
        plt.xlabel("Sample Index")
        plt.ylabel("ltv_sum")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Indicator evaluation
def evaluate_ltv(preds_results, cycles=pre_cycles):
    """
    return RMSE, MAE, MSLE, R2
    user sample
    days_list_existed: have existed days for valid
    preds_results: the predicted results by running the trained model
    """
    eval_dict = {}
    for i in range(cycles):
        y_true = preds_results[i]["actual"].values
        y_pred = preds_results[i]["pred"].values
        rmse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        msle = mean_squared_log_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        eval_dict[day] = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MSLE": round(msle, 4),
            "R2": round(r2, 4),
        }
        return eval_dict


# Residual analysis
def residual_plot(preds_results, cycles=pre_cycles):
    """
    Objective: show the residuals of LTV
    days_list_existed: have existed days for valid
    preds_results: the predicted results by running the trained model
    """
    days_list_existed = config["days_list_existed"]
    for i in range(cycles):
        residuals = (
            preds_results[i]["pred"].values - preds_results[i]["actual"].values
        )
        plt.figure(figsize=(20, 4))
        plt.plot(residuals, marker="o", linestyle="-", color="purple", alpha=0.7)
        plt.title(f"Residuals Over Samples--Month Cycle--{i+1}")
        plt.xlabel("Sample Index")
        plt.ylim(-5000, 5000)
        plt.ylabel("Residual (Prediction Error)")
        plt.grid(True)
        plt.show()
