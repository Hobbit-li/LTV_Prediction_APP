"""
Data Utilities Module

Contains helper functions for data processing:
- Data splitting
- Feature engineering
- Preprocessing pipelines
"""

# import pandas as pd
# from config_loader import load_configf
from sklearn.model_selection import train_test_split

# config = load_config()


def data_preprocess(df, ref_month, config: dict, train_data=True):
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
    # ref_month = config["ref_month"]
    num_features = config["num_features_map"][ref_month]
    cat_features = config["cat_features"]
    target_col = config["target_col_map"][ref_month]
    id_col = config["id_col"]

    x_df = df[num_features + cat_features]
    y_df = df[target_col]
    user_ids = df[id_col]

    # transform type: category
    for col in cat_features:
        x_df[col] = x_df[col].astype("category")

    if train_data:
        x_train, x_valid, y_train, y_valid, id_train, id_valid = train_test_split(
            x_df, y_df, user_ids, test_size=0.3, random_state=42
        )
    else:
        x_train = x_valid = x_df
        y_train = y_valid = y_df
        id_train = id_valid = user_ids

    # Store outputs
    result = {"train": {}, "valid": {}}
      # Split into payer vs non-payer
    x_train_1, x_train_2, y_train_1, y_train_2 = paid_split(
            x_train, y_train, config
        )
    x_valid_1, x_valid_2, y_valid_1, y_valid_2 = paid_split(
            x_valid, y_valid, config
        )

        # Store all sets in dict
    result["train"] = {
            "all": (x_train, y_train, id_train),
            "nonpayer": (x_train_1, y_train_1),
            "payer": (x_train_2, y_train_2),
        }

    result["valid"] = {
            "all": (x_valid, y_valid, id_valid),
            "nonpayer": (x_valid_1, y_valid_1),
            "payer": (x_valid_2, y_valid_2),
        }

    return result


def paid_split(x_df, y_df, config: dict):
    """
    Split data into payer/non-payer subsets

    Args:
        data (pd.DataFrame): Input dataset
        payer_tag (str): Column name indicating payer status
    Returns:
        tuple: Split datasets
    """
    payer_tag = config["payer_tag"]

    existing_payer_tag = [col for col in payer_tag if col in x_df.columns]
    if not existing_payer_tag:
        raise ValueError(
            "=None of the payer_tag columns are present in x_df, unable to identify unpaid users."
        )

    # Unpaid samples during the feature period (Set 1)
    mask_unpaid = x_df[existing_payer_tag].sum(axis=1) == 0
    x1 = x_df[mask_unpaid]
    y1 = y_df[mask_unpaid]

    # Paid samples during the feature period (Set 2)
    mask_paid = ~mask_unpaid
    x2 = x_df[mask_paid]
    y2 = y_df[mask_paid]

    return x1, x2, y1, y2
