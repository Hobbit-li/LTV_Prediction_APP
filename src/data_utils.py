import pandas as pd
from config_loader import load_config

config = load_config()
payer_tag = config["payer_tag"]


def paid_split(X, y, payer_tag=payer_tag):
    """
    split the users
    set 1: HAVE Unpaid until the current period
    set 2: HAVE Paid until the peroid
    """
    existing_payer_tag = [col for col in payer_tag if col in X.columns]
    if not existing_payer_tag:
        raise ValueError(
            "❌ None of the payer_tag columns are present in X, unable to identify unpaid users."
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
