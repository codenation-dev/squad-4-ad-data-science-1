import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def dummies(df, drop_first = True):
    categorical_features = df.select_dtypes("object").columns
    dummy = pd.get_dummies(df[categorical_features], drop_first = drop_first)
    df = df.merge(dummy, left_index = True, right_index = True)

    return df

def numerical_scaler(df, cols):
    scaler= StandardScaler()

    df[cols] = scaler.fit_transform(df[cols])
    return df
