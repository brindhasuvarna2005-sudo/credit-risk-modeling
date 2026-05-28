from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def select_features(X_train, y_train, X_test, top_n=50):

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    })

    feature_importance = feature_importance.sort_values(
        by="importance", ascending=False
    )

    selected_features = feature_importance.head(top_n)["feature"].tolist()

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    return X_train_selected, X_test_selected, selected_features