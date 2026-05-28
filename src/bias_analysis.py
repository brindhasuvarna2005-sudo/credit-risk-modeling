import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


# ==============================
# GENERIC FAIRNESS ANALYSIS
# ==============================
def group_bias_analysis(model, X_full, X_model, y_test,
                       group_col, threshold=0.2):

    results = []

    # Align indices
    X_full = X_full.reset_index(drop=True)
    X_model = X_model.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Remove NaN groups
    valid_mask = X_full[group_col].notna()
    X_full = X_full[valid_mask]
    X_model = X_model[valid_mask]
    y_test = y_test[valid_mask]

    for group in X_full[group_col].unique():

        idx = X_full[X_full[group_col] == group].index

        if len(idx) == 0:
            continue

        X_group = X_model.loc[idx]
        y_true = y_test.loc[idx]

        # Predictions
        y_prob = model.predict_proba(X_group)[:, 1]
        y_pred = (y_prob > threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        results.append({
            "group": str(group),
            "samples": len(idx),
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "positive_rate": np.mean(y_pred),
            "TPR (recall)": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0
        })

    return pd.DataFrame(results)


# ==============================
# GENDER BIAS ANALYSIS
# ==============================
def gender_bias_analysis(model, X_full, X_model, y_test,
                         threshold=0.2):

    results = []

    # Detect gender columns
    gender_cols = [c for c in X_full.columns if "GENDER" in c]

    if len(gender_cols) == 0:
        raise ValueError("No gender columns found")

    # Align indices
    X_full = X_full.reset_index(drop=True)
    X_model = X_model.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    for col in gender_cols:
        for val in [0, 1]:

            idx = X_full[X_full[col] == val].index

            if len(idx) == 0:
                continue

            X_group = X_model.loc[idx]
            y_true = y_test.loc[idx]

            y_prob = model.predict_proba(X_group)[:, 1]
            y_pred = (y_prob > threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()

            results.append({
                "group": f"{col}={val}",
                "samples": len(idx),
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "positive_rate": np.mean(y_pred),
                "TPR (recall)": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0
            })

    return pd.DataFrame(results)


# ==============================
# FAIRNESS GAP FUNCTION
# ==============================
def fairness_gap(df, metric):
    if metric not in df.columns:
        raise ValueError(f"{metric} not found in DataFrame. Available: {df.columns.tolist()}")
    return df[metric].max() - df[metric].min()