# Predict and save results
# result_df = pd.DataFrame({
#     'user_id': id_test.values
import pandas as pd      # DataFrame 操作、concat、copy 等
import numpy as np       # np.expm1、np.clip
import lightgbm as lgb

from config_loader import load_config
config = load_config()
payer_tag = config["payer_tag"]

def predict_process(X1, X2, y1, y2, model1, model2, payer_tag=payer_tag):
    '''
        Classification model prediction.
        Extract groups with potential to pay.
        Regression model predicts LTV.
    '''
    existing_payer_tag = [col for col in payer_tag if col in X1.columns]
    if not existing_payer_tag:
        raise ValueError("❌ No payer_tag columns found in X; unable to identify unpaid users.")
    # print(eval(f"X_test_day{day}_1").drop(columns=existing_payer_tag).head())

    temp = model1.predict(X1.drop(columns=existing_payer_tag), num_iteration=model1.best_iteration)
    
    # Deep copy
    X1_copy = X1.copy()
    X1_copy['pay_class_pred'] = (temp > 0.5).astype(int)
   
    # Filter data
    mask_payfu = (X1_copy['pay_class_pred'] == 1)
    X_combined = pd.concat(
        [X1_copy[mask_payfu].drop(columns='pay_class_pred'), X2], axis=0)
    y_combined = pd.concat(
        [y1[mask_payfu], y2], axis=0)

    preds_log = model2.predict(X_combined, num_iteration=model2.best_iteration)
    preds = pd.Series(np.expm1(preds_log)).clip(lower=0).values
    
    X_combined['actual'] = y_combined.values
    X_combined['pred'] = preds
    # print(eval(f"X_combined_test_day{day}").head())

    # Add information for players predicted not to pay in the next step
    X1_copy.loc[~mask_payfu, 'actual'] = y1[~mask_payfu].values
    X1_copy.loc[~mask_payfu, 'pred'] = 0
    # print(y1[~mask_payfu].values)
    # print(X1_copy[~mask_payfu][['actual', 'pred']].head())

    temp = pd.concat([X_combined, X1_copy[~mask_payfu].drop(columns='pay_class_pred')], axis=0)

    pred_results = pd.concat([id_test.rename('uid'), temp], axis=1)

    return pred_results
