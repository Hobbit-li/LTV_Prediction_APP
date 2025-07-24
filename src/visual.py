# visualization
# plot
# model evaluate

# existing life cycles for valid
def compara_plot(days_list_existed=days_list_existed, preds_results):
  '''
    Objective: show the actual and the predicted LTV increasingly
    days_list_existed: have existed days for valid
    preds_results: the predicted results by running the trained model
  '''
  for day in days_list_existed:
    plt.figure(figsize=(20, 6))
    result_df_sorted = preds_results[day].sort_values(by="actual").reset_index(drop=True)
    plt.plot(result_df_sorted["actual"], label='Actual', color='blue', linewidth=2)
    plt.plot(result_df_sorted["pred"], label='Predicted', color='orange', linewidth=2, linestyle='--')
    
    plt.title(f"Model Prediction vs Actual--DAY{day}")
    plt.xlabel('Sample Index')
    plt.ylabel('LTV')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Indicator evaluation
def evaluate_ltv(days_list_existed=days_list_existed, preds_results):
  '''
    return RMSE, MAE, MSLE, R2
    user sample
    days_list_existed: have existed days for valid
    preds_results: the predicted results by running the trained model
  '''
  
  eval_dict = {}
  for day in days_list_existed:
    y_true = preds_results[day]['actual'].values
    y_pred = preds_results[day]['pred'].values
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    eval_dict[day] = {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'MSLE': round(msle, 4),
            'R2': round(r2, 4)
        }
    return eval_dict

# Residual analysis
def compara_plot(days_list_existed=days_list_existed, preds_results):
  '''
    Objective: show the residuals of LTV
    days_list_existed: have existed days for valid
    preds_results: the predicted results by running the trained model
  '''
  for day in days_list_existed:
    residuals = preds_results[day]["pred"].values - preds_results[day]["actual"].values
    plt.figure(figsize=(20, 4))
    plt.plot(residuals, marker='o', linestyle='-', color='purple', alpha=0.7)
    plt.title(f"Residuals Over Samples--DAY{day}")
    plt.xlabel('Sample Index')
    plt.ylim(-5000,5000)
    plt.ylabel('Residual (Prediction Error)')
    plt.grid(True)
    plt.show()

    
      
      
    
     
    
