import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    columns_to_drop = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df

def split_dataset(df, feature_names):
    X = df[feature_names]
    y = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    return train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )