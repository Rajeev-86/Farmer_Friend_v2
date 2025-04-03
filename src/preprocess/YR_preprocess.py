import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def fit_preprocessing(X_train):
    """Fits OneHotEncoder and StandardScaler on training data."""
    
    # Using sin-cos transformation for 'Month' (makes transitions smooth)
    X_train = X_train.copy()
    X_train['Month_sin'] = np.sin(2 * np.pi * X_train['Month'] / 12)
    X_train['Month_cos'] = np.cos(2 * np.pi * X_train['Month'] / 12)
    X_train.drop(columns=['Month'], inplace=True)

    # Converting 'Year' to relative years since 2015
    X_train['Year'] = X_train['Year'] - 2015

    # Identify categorical & numerical columns
    cat_columns = ['Crop', 'Region']
    num_columns = X_train.drop(columns=cat_columns).columns

    # Initialize encoders
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    scaler = StandardScaler()

    # Fit encoders on training data
    ohe.fit(X_train[cat_columns])
    scaler.fit(X_train[num_columns])

    return ohe, scaler  # Return fitted encoders

def transform_preprocessing(X, ohe, scaler):
    """Applies fitted OneHotEncoder and StandardScaler to new data."""
    
    X = X.copy()

    # Apply same transformations as training set
    X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
    X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
    X.drop(columns=['Month'], inplace=True)

    X['Year'] = X['Year'] - 2015

    cat_columns = ['Crop', 'Region']
    num_columns = X.drop(columns=cat_columns).columns

    # Transform using fitted encoders
    cat_data = ohe.transform(X[cat_columns])
    num_data = scaler.transform(X[num_columns])

    # Convert categorical data to DataFrame
    cat_data = pd.DataFrame(cat_data, columns=ohe.get_feature_names_out(cat_columns))
    num_data = pd.DataFrame(num_data, columns=num_columns)

    # Reset index to avoid mismatches during concatenation
    X_transformed = pd.concat([num_data.reset_index(drop=True), cat_data.reset_index(drop=True)], axis=1)

    return X_transformed
