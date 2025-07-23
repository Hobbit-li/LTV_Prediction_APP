# Build a classification model - Predict whether a player will pay or not
# Essentially, predict whether players who have not paid previously will make a breakthrough payment in the subsequent prediction period
# Since the number of paid users is very small, increase the weight of positive samples

def train_clf(X_train, X_valid, y_train, y_valid, params_clf=params_clf, payer_tag=payer_tag):
    '''
        Dataset: Users who have not paid during the feature period
        Binary classification: Predict whether a breakthrough payment will occur in the subsequent period, 0/1
        return: Model and model performance evaluation
    '''
    existing_payer_tag = [col for col in payer_tag if col in X_train.columns]
    if not existing_payer_tag:
        raise ValueError("❌ None of the payer_tag columns are present in X, unable to identify unpaid users.")
        
    # Feature processing
    # Keep only active dimension features
    X_tr = X_train.drop(columns=existing_payer_tag)
    X_val = X_valid.drop(columns=existing_payer_tag)
    y_tr = (y_train > 0).astype(int)
    y_val = (y_valid > 0).astype(int)
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val)
    
    clf = lgb.train(
        params_clf,
        train_set,
        valid_sets=[train_set, val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = list(X_tr.columns)
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    fold_importance_df = fold_importance_df.sort_values(by='importance', ascending=False)
    # fold_importance_df.to_csv(f"importance_df_clf.csv", index=None) 

    # print("Example of input feature matrix for classification prediction:", X_val)
    y_pred_proba = clf.predict(X_val, num_iteration=clf.best_iteration)
    # Use probability threshold for binary classification
    y_pred = (y_pred_proba > 0.5).astype(int)

    result = {
        "classification_report": classification_report(y_val, y_pred),
        "AUC": roc_auc_score(y_val, y_pred_proba)
    }

    return clf, result, fold_importance_df


# Wrapper -- Training and prediction for the model
def train_reg(
    X_train, X_valid, y_train, y_valid, params, num_features, cat_features, 
    value_weighting=True):
    
    '''
    Training function
    - value_weighting: Whether to apply weights for high-value users. If weighting is needed, set this parameter to True.
    - quantile: High-value quantile threshold (e.g., top 1%), which should be adjusted according to different projects.
    - weight_multiplier: Weight multiplier for high-value users.
    '''
    
    # Apply log1p transformation to target
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # Default weights are 1
    train_weights = np.ones(len(y_train))

    if value_weighting:
        bins = sorted(set([y_train.quantile(p) for p in percentiles]))

        if len(bins) < 2:
            raise ValueError("Insufficient valid quantile bins, possibly all y_train values are 0 or duplicates.")
    
        labels = list(range(len(bins) - 1))
        quantile_bins = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
    
        # Generate default weights of 1
        train_weights = np.ones(len(y_train))
        # Only apply weights to successfully binned samples
        mask_valid = ~quantile_bins.isna()
        quantile_ids = quantile_bins[mask_valid].astype(int)
        train_weights[mask_valid] = [base_weights[i] for i in quantile_ids]
        # Manually set the largest values (e.g., top 10 samples) to 1000
        top_k_idx = np.argsort(y_train.values)[-top_num:]  # Top 10 largest samples
        train_weights[top_k_idx] = base_weights[-1]  # Force the value to 1000

    print(pd.value_counts(train_weights, sort=False))

        # high_value_threshold = np.quantile(y_train, quantile)
        # high_value_mask = y_train >= high_value_threshold
        # train_weights[high_value_mask] = weight_multiplier
        # print(f"⚖️ Weighted high-value users: {high_value_mask.sum()} samples have their weight set to {weight_multiplier}")

    features = num_features + cat_features
    
    # Wrap dataset
    trn_data = lgb.Dataset(X_train, label=y_train_log, categorical_feature=cat_features, weight=train_weights)
    val_data = lgb.Dataset(X_valid, label=y_valid_log, categorical_feature=cat_features, reference=trn_data)
    reg = lgb.train(params_reg,
                    trn_data,
                    valid_sets=[trn_data, val_data],
                    feval = r2_eval,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(period=500)])  # Equivalent to verbose_eval=100

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = reg.feature_importance()
    fold_importance_df["importance_gain"] = reg.feature_importance(importance_type='gain')
    fold_importance_df = fold_importance_df.sort_values(by='importance', ascending=False)
    # fold_importance_df.to_csv(f"importance_df_reg.csv", index=None) 

    # print("Example of input feature matrix for regression prediction:", X_valid)
    # Predict log values
    y_preds_log = reg.predict(X_valid, num_iteration=reg.best_iteration)
    # Restore log and correct negative values
    y_preds = pd.Series(np.expm1(y_preds_log)).clip(lower=0).values

    # Model evaluation metrics -- Mean Squared Log Error and R2 Score
    result = {
        "MSLE": mean_squared_log_error(y_valid, y_preds),
        "R2": r2_score(y_valid, y_preds)
    }
    
    return reg, result, fold_importance_df


# Training process wrapper
# Fixed prediction period
# Compatible with empty validation set
def train_process(X_train_1, X_valid_1, X_train_2, X_valid_2, y_train_1, y_valid_1, y_train_2, y_valid_2, params_clf=params_clf, params_reg=params_reg, num_features, cat_features=cat_features, payer_tag=payer_tag):
    '''
    The binary classification model determines future payment behavior.
    The regression model predicts future LTV.
    _1: Dataset of players who have not paid during the feature period
    _2: Dataset of players who have paid (payer) during the feature period
    '''
    # Check if the validation set is empty
    # The purpose is to train only, without validation
    only_train = any(df is None for df in [X_valid_1, X_valid_2, y_valid_1, y_valid_2])
    if only_train:
        # Construct empty DataFrames with the same structure
        X_valid_1 = pd.DataFrame(columns=X_train_1.columns)
        X_valid_2 = pd.DataFrame(columns=X_train_2.columns)
        y_valid_1 = pd.Series(dtype='float64')
        y_valid_2 = pd.Series(dtype='float64')
        print("Train only, no validation")

    # Train the classification model on the dataset of players who have not paid during the feature period
    clf_valid, result_valid_clf, importance_clf = train_clf(X_train_1, X_valid_1, y_train_1, y_valid_1, params_clf)
    
    # Predict on dataset 1
    existing_payer_tag = [col for col in payer_tag if col in X_train_1.columns]
    if not existing_payer_tag:
        raise ValueError("❌ None of the payer_tag columns are present in X, unable to identify unpaid users.")
    temp = clf_valid.predict(X_train_1.drop(columns=existing_payer_tag))
    
    # Create deep copy
    X_train_1_copy = X_train_1.copy()
    X_train_1_copy['pay_class_pred'] = (temp > 0.5).astype(int)
    
    # eval(f"X_train_day{day}_1")['pay_class_pred'] = (temp > 0.5).astype(int)
    # print(eval(f"X_train_day{day}_1").drop(columns=existing_payer_tag).head())
    
    # Create deep copy
    X_valid_1_copy = X_valid_1.copy()
    if not X_valid_1.empty:
        temp = clf_valid.predict(X_valid_1.drop(columns=existing_payer_tag))
        X_valid_1_copy['pay_class_pred'] = (temp > 0.5).astype(int)
    else:
        X_valid_1_copy['pay_class_pred'] = pd.Series(dtype='int')
    # eval(f"X_valid_day{day}_1")['pay_class_pred'] = (temp > 0.5).astype(int)

    # Filter data
    mask_payfu_1 = (X_train_1_copy['pay_class_pred'] == 1)
    mask_payfu_2 = (X_valid_1_copy['pay_class_pred'] == 1)

    # Concatenate data
    X_combined_train = pd.concat(
        [X_train_1_copy[mask_payfu_1].drop(columns='pay_class_pred'), X_train_2], axis=0)
    y_combined_train = pd.concat(
        [y_train_1[mask_payfu_1], y_train_2], axis=0)

    X_combined_valid = pd.concat(
        [X_valid_1_copy[mask_payfu_2].drop(columns='pay_class_pred'), X_valid_2], axis=0)
    y_combined_valid = pd.concat(
        [y_valid_1[mask_payfu_2], y_valid_2], axis=0)

    reg_valid, result_valid_reg, importance_reg = train_reg(X_combined_train, X_combined_valid, y_combined_train, y_combined_valid, params_reg, num_features, cat_features)
    model_result = {
        'model_clf': clf_valid,
        'result_clf': result_valid_clf,
        'im_clf': importance_clf,
        'model_reg': reg_valid,
        'result_reg': result_valid_reg,
        'im_reg': importance_reg
    }

    # If only training the model without validation
    # Only return the models
    if only_train:
        model_result.update({
            'model_clf': clf_valid,
            'model_reg': reg_valid,
        })
    return model_result
