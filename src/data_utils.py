def paid_split(X, y, payer_tag):
    existing_payer_tag = [col for col in payer_tag if col in X.columns]
    if not existing_payer_tag:
        raise ValueError("❌ X中不包含任何payer_tag列，无法识别未付费用户。")
    # 特征周期内未付费样本（集合1）
    mask_unpaid = (X[existing_payer_tag].sum(axis=1) == 0)
    X1 = X[mask_unpaid]
    y1 = y[mask_unpaid] 

    # 特征周期内已付费样本（集合2）
    mask_paid = ~mask_unpaid
    X2 = X[mask_paid]
    y2 = y[mask_paid]

    return X1, X2, y1, y2
