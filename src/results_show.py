"""
Results Presentation Module

Contains functions for displaying and analyzing model results
"""


def show_roas_ltv(preds_results, cost, cycles=pre_cycles, payer_tag=payer_tag):
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
    result = {}
    ltv_existed = preds_results[payer_tag].sum(axis=1)
    y_pred = preds_results[0]["pred"]
    y_actual = preds_results[0]["actual"]
    for i in range(cycles):
        # df_temp = preds_results[i]
        roas_pred = y_pred.sum() + ltv_existed.sum() / cost
        roas_actual = y_actual.sum() + ltv_existed.sum() / cost
        ltv_pred = (y_pred + ltv_existed).mean()
        ltv_actual = (y_actual + ltv_existed).mean()

        result[day] = {
            "ROAS_pred": roas_pred,
            "ROAS_actual": roas_actual,
            "LTV_pred": ltv_pred,
            "LTV_actual": ltv_actual,
        }
        try:
            y_pred = y_pred + preds_results[i+1]["pred"]
            y_actual = y_actual + preds_results[i+1]["actual"]
        except (IndexError, KeyError):
            raise IndexError(f"index i={i+1} not exists in preds_results.")
        

    return result
