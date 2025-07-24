import os
import json
import joblib
import pandas as pd
from datetime import datetime

def create_output_dir(base_dir="outputs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_predictions(preds_results, output_dir):
    for day, df in preds_results.items():
        df.to_csv(os.path.join(output_dir, f"preds_day{day}.csv"), index=False)

def save_metrics(metrics_dict, output_dir):
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    # optional csv
    pd.DataFrame.from_dict(metrics_dict, orient="index").to_csv(
        os.path.join(output_dir, "metrics.csv")
    )

def save_model(model, output_dir, name="model.pkl"):
    joblib.dump(model, os.path.join(output_dir, name))
