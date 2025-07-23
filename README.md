
# LTV Predictor Pipeline

A lightweight pipeline to predict multi-day LTV using classification + regression hybrid model.

## Files
- `main.py` - Run the full pipeline
- `train.py` - Training logic for classifier & regressor
- `predict.py` - Inference logic
- `data_utils.py` - Data loading and processing
- `config.py` - Feature and label config

## Usage

1. Put your CSV file locally (see `main.py`)
2. Run:

```bash
python main.py
