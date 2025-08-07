"""
Data Utilities Module

Contains helper functions for data processing:
- Data splitting
- Feature engineering
- Preprocessing pipelines
"""

from sklearn.model_selection import train_test_split

def data_preprocess(df, config:dict, ref_month='m5', train_if=True):
    """
    Preprocess data and split into payer/non-payer groups for each prediction day.

    - Parameters:
        - df (pd.DataFrame): Input DataFrame with features and targets
        - config (dict): 
            - num_features/cat_features (list): Numerical/categorical feature names
            - target_col (list): Target column names
            - id_col (str): User ID column
            - payer_tag (list[str]): Column names indicating payer status
        - ref_month (str): Key of the refrence month
        - train_if (bool): If True, split data into train/valid sets, default: True
        
    - Returns:
        - Dict with 'train'/'valid' keys, each containing:
            {
                "all": (features, target, unique_id),
                "nonpayer": (features, target),
                "payer": (features, target)
            }
        -  target_num (int): Numbers of predicted cycles
            
    """
    num_features = config["num_features_map"][ref_month]
    cat_features = config["cat_features"]
    target_col = config["target_col_map"][ref_month]
    id_col = config["id_col"]
    payer_tag = config["payer_tag"]

    x_df = df[num_features + cat_features]
    y_df = df[target_col]
    user_ids = df[id_col]
    
    target_num = y_df.shape[1]
    # transform type: category
    for col in cat_features:
        x_df[col] = x_df[col].astype("category")

    if train_if:
        x_train, x_valid, y_train, y_valid, id_train, id_valid = train_test_split(
            x_df, y_df, user_ids, test_size=0.3, random_state=42
        )
    else:
        x_train = x_valid = x_df
        y_train = y_valid = y_df
        id_train = id_valid = user_ids

    result = {"train": {}, "valid": {}}
    # Split into payer vs non-payer
    x_train_1, x_train_2, y_train_1, y_train_2 = paid_split(
            x_train, y_train, payer_tag
        )
    x_valid_1, x_valid_2, y_valid_1, y_valid_2 = paid_split(
            x_valid, y_valid, payer_tag
        )
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

    return result, target_num


def paid_split(x_df, y_df, payer_tag):
    """
    Split data into payer/non-payer subsets

    - Parameters:
        - x_df (pd.DataFrame): Input dataset including features
        - y_df (pd.DataFrame): Input dataset including targets
        - payer_tag (list[str]): Column names indicating payer status
    
    - Returns (tuple): 4 dataframes, Split datasets
    """
    
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
