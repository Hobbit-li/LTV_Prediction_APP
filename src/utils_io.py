"""
I/O Utilities Module

Provides file operations for:
- Model serialization
- Data saving/loading
- Result storage
"""

import os
from datetime import datetime
import joblib
import pandas as pd


def create_output_dir(base_dir="outputs"):
    """
    Creates a timestamped output directory for storing results.

    Args:
        base_dir (str, optional): The base directory where output folder will be created.
        Defaults to "outputs".

    Returns:
        str: Path to the created output directory (format: base_dir/run_YYYYMMDD_HHMMSS)
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_predictions(preds_results, output_dir):
    """
    Saves prediction results to CSV files in the output directory.

    Args:
        preds_results (dict): Dictionary containing DataFrames of predictions,
        where keys are day numbers (e.g., {1: df_day1, 2: df_day2})
        output_dir (str): Path to the directory where prediction files will be saved
    """
    output_dir = Path(output_dir)
    for day, df in preds_results.items():
        df.to_csv(output_dir / f"preds_day{day}.csv", index=False)


def save_metrics(metrics_dict, output_dir):
    """
    Saves evaluation metrics to JSON and optionally to CSV format.

    Args:
        metrics_dict (dict): Dictionary containing evaluation metrics
        output_dir (str): Path to the directory where metrics files will be saved
    """
    # json_path = os.path.join(output_dir, "metrics.json")
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(metrics_dict, f, indent=2)

    # optional csv
    output_dir = Path(output_dir)
    pd.DataFrame.from_dict(metrics_dict, orient="index").to_csv(
        output_dir / "metrics.csv"
    )


def save_model(model, output_dir, name="model.pkl"):
    """
    Serialize and save model to disk

    Args:
        model: Trained model object
        output_dir (str): Output file path
    """
    output_dir = Path(output_dir)
    joblib.dump(model, output_dir / name)
