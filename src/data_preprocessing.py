import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df, target_col="TARGET", id_col="SK_ID_CURR", test_size=0.2, random_state=42):

    # Drop ID column
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test