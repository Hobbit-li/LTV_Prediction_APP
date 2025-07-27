"""
Results Presentation Module

Contains functions for displaying and analyzing model results
"""

def show_roas_ltv(preds_results, config):
    """
    Evaluate predicted vs. actual ROAS and LTV values.

    Parameters:
    - preds_results (dict): {day: DataFrame with 'pred' and 'actual' columns}
    - days_list (list[int]): Days to evaluate (e.g., [7, 30, 60])
    - cost (float): Total cost for ROAS calculation

    Returns:
    - result_dict (dict): {
          day: {
              'ROAS_pred': float,
              'ROAS_actual': float,
              'LTV_pred': float,
              'LTV_actual': float
          }, ...
      }
    """
    days_list = config["days_list"]
    cost = config["cost"]
    result = {}

    for day in days_list:
        df_temp = preds_results[day]
        roas_pred = df_temp["pred"].sum() / cost
        roas_actual = df_temp["actual"].sum() / cost
        ltv_pred = df_temp["pred"].mean()
        ltv_actual = df_temp["actual"].mean()

        result[day] = {
            "ROAS_pred": roas_pred,
            "ROAS_actual": roas_actual,
            "LTV_pred": ltv_pred,
            "LTV_actual": ltv_actual,
        }

    return result
