import pandas as pd
# from config_loader import load_config
from sklearn.model_selection import train_test_split

# config = load_config()

"""
Data Utilities Module

Contains helper functions for data processing:
- Data splitting
- Feature engineering
- Preprocessing pipelines
"""


def data_preprocess(
    df,
    config:dict,
    train_data=True
):
    """
    Preprocess data and split into payer/non-payer groups for each prediction day.
    
    Args:
        df: Input DataFrame with features and targets
        train_data: If True, split data into train/valid sets (default True)
        config: Dict containing:
            - payer_tag: Payer indicator column
            - num_features/cat_features: Numerical/categorical feature names  
            - target_col: Target column names
            - id_col: User ID column  
            - days_list: Prediction days to process
            - num_features_map: Features for each day
    
    Returns:
        Dict with 'train'/'valid' keys, each containing day-wise splits:
        {
            day: {
                "all": (features, target),
                "nonpayer": (features, target), 
                "payer": (features, target)
            }
        }
    """

    payer_tag = config["payer_tag"]
    num_features = config["num_features"]
    cat_features = config["cat_features"]
    target_col = config["target_col"]
    id_col = config["id_col"]
    days_list = config["days_list"]
    num_features_map = config["num_features_map"]


    X = df[num_features + cat_features]
    y = dfconfig[target_col]
    user_ids = df[id_col]

    # transform type: category
    for col in cat_features:
        X[col] = X[col].astype("category")

    if train_data:
        X_train, X_valid, y_train, y_valid, id_train, id_valid = train_test_split(
            X, y, user_ids, test_size=0.3, random_state=42
        )
    else:
        X_train = X_valid = X
        y_train = y_valid = y
        id_train = id_valid = user_ids

    # Store outputs
    result = {"train": {}, "valid": {}}

    for i, day in enumerate(days_list):
        features = num_features_map[day] + cat_features

        # Get feature and target subsets
        X_train_day = X_train[features]
        X_valid_day = X_valid[features]
        y_train_day = y_train.iloc[:, i]
        y_valid_day = y_valid.iloc[:, i]

        # Split into payer vs non-payer
        X_train_day_1, X_train_day_2, y_train_day_1, y_train_day_2 = paid_split(
            X_train_day, y_train_day, payer_tag
        )
        X_valid_day_1, X_valid_day_2, y_valid_day_1, y_valid_day_2 = paid_split(
            X_valid_day, y_valid_day, payer_tag
        )

        # Store all sets in dict
        result["train"][day] = {
            "all": (X_train_day, y_train_day),
            "nonpayer": (X_train_day_1, y_train_day_1),
            "payer": (X_train_day_2, y_train_day_2),
        }

        result["valid"][day] = {
            "all": (X_valid_day, y_valid_day),
            "nonpayer": (X_valid_day_1, y_valid_day_1),
            "payer": (X_valid_day_2, y_valid_day_2),
        }

    return result


def paid_split(X, y, config:dict):
    """
    Split data into payer/non-payer subsets
    
    Args:
        data (pd.DataFrame): Input dataset
        payer_tag (str): Column name indicating payer status
        days_list (list): Time periods for analysis
        num_features (list): Numerical features
        cat_features (list): Categorical features
        target_col (str): Target variable name
        id_col (str): Identifier column name
        num_features_map (dict): Numerical feature metadata
        
    Returns:
        tuple: Split datasets
    """
    payer_tag = config["payer_tag"]

    existing_payer_tag = [col for col in payer_tag if col in X.columns]
    if not existing_payer_tag:
        raise ValueError(
            "=None of the payer_tag columns are present in X, unable to identify unpaid users."
        )

    # Unpaid samples during the feature period (Set 1)
    mask_unpaid = X[existing_payer_tag].sum(axis=1) == 0
    X1 = X[mask_unpaid]
    y1 = y[mask_unpaid]

    # Paid samples during the feature period (Set 2)
    mask_paid = ~mask_unpaid
    X2 = X[mask_paid]
    y2 = y[mask_paid]

    return X1, X2, y1, y2
