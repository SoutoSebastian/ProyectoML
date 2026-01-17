import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """

    return pd.read_csv(path, sep = ';')


def split_features_target(df: pd.DataFrame, target_col: str ):
    """
    Split dataframe into features(X) and target (y).
    """

    X = df.drop(columns=["y"])
    y = df[target_col].map({'yes' : 1, 'no' : 0})

    return X, y


def get_column_types(X:pd.DataFrame):
    """
    Identify numerical and categorical columns.
    """

    num_cols = X.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include = ['object']).columns.to_list()

    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    """
    Build a ColumnTransformer for preprocessing numerical and categorical features.
    """

    preprocessor = ColumnTransformer(
        transformers = [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return preprocessor


def prepare_features(df: pd.DataFrame, target_col:str):
    """
    Prepare features and preprocessing pipeline.
    """

    X, y = split_features_target(df, target_col)
    num_cols, cat_cols = get_column_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    return X, y, preprocessor