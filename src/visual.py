"""
Visualization Module

Provides visualization tools for model evaluation including:
- Performance comparison plots
- LTV evaluation charts
- Residual analysis plots
"""
import base64
import io
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
)


def compare_plot(preds_results, cycles=10):
    """
    Show the actual and the predicted LTV increasingly

    - parameters:
        - preds_results: Predicted results by running the trained model
        - cycles: Numbers of predicted cycles, default: 10

    - return: figs
    """
    figs = []
    for i in range(cycles):
        fig, ax = plt.subplots(figsize=(20, 6))
        result_df_sorted = (
            preds_results[i].sort_values(by="actual").reset_index(drop=True)
        )
        ax.plot(result_df_sorted["actual"], label="Actual", color="blue", linewidth=2)
        ax.plot(
            result_df_sorted["pred"],
            label="Predicted",
            color="orange",
            linewidth=2,
            linestyle="--",
        )

        ax.set_title(f"Model Prediction vs Actual -- Month Cycle -- {i+1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("ltv_sum")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        figs.append(fig)

    return figs

# Residual analysis
def residual_plot(preds_results, cycles=10):
    """
    Show the residuals between actual and the predicted LTV

    - parameters:
        - preds_results: Predicted results by running the trained model
        - cycles: Numbers of predicted cycles, default: 10

    - return: figs
    """
    figs = []
    for i in range(cycles):
        residuals = preds_results[i]["pred"].values - preds_results[i]["actual"].values

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(residuals, marker="o", linestyle="-", color="purple", alpha=0.7)
        ax.set_title(f"Residuals Over Samples -- Month Cycle -- {i+1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Residual (Prediction Error)")
        ax.set_ylim(-5000, 5000)
        ax.grid(True)
        fig.tight_layout()

        figs.append(fig)

    return figs

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
