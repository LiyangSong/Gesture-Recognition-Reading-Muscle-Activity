import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def identify_missing_attrs(df: pd.DataFrame, missing_threshold: float = 0.20) -> list:
    missing_attrs = []
    for attr in df.columns:
        if np.mean(df[attr].isna()) >= missing_threshold:
            missing_attrs.append(attr)
    return missing_attrs


def split_numerical_nominal(df: pd.DataFrame) -> (list, list):
    numerical_attrs = df.select_dtypes(include=['number']).columns.tolist()
    nominal_attrs = df.select_dtypes(exclude=['number']).columns.tolist()
    return numerical_attrs, nominal_attrs


def get_default_preprocessor(numerical_attrs: list, nominal_attrs: list) -> any:
    numerical_transformer = Pipeline([
        ('simple_imputer', SimpleImputer()),
        ('standard_scaler', StandardScaler())
    ])

    # Use OneHot Encoder to transform nominal attrs.
    # Target Encoder cannot be used as this is unsupervised.
    nominal_transformer = Pipeline([
        ('simple_imputer', SimpleImputer(strategy='most_frequent')),
        ('oneHot_encoder', OneHotEncoder()),
        ('standard_scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_transformer, numerical_attrs),
        ("nominal", nominal_transformer, nominal_attrs)
    ])

    return preprocessor
