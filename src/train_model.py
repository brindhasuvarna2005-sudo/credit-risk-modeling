from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# ==============================
# LOGISTIC REGRESSION (PIPELINE )
# ==============================
def train_logistic_regression(X_train, y_train, sample_weight=None):

    model = Pipeline([
        ("scaler", StandardScaler()),   # scaling inside pipeline
        ("classifier", LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced"
        ))
    ])

    # Pass sample weights correctly to classifier inside pipeline
    if sample_weight is not None:
        model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    return model   # ONLY model (pipeline)


# ==============================
# XGBOOST
# ==============================
def train_xgboost(X_train, y_train, sample_weight=None):

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    return model   #  ONLY model