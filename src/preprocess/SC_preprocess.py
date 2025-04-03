from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

def fit_preprocessor(X_train):
    # Initialize the encoders
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    scaler = StandardScaler()

    # Fit the one-hot encoder on categorical column
    ohe.fit(X_train[['Crop_Type', 'Soil_Type']])

    # Fit the scaler on numerical columns
    num_cols = X_train.select_dtypes(include=['number']).columns
    scaler.fit(X_train[num_cols])

    return ohe, scaler  # Return both fitted transformers

def transform_preprocessor(X, ohe, scaler):
    # One-hot encoding for categorical column
    X_encoded = ohe.transform(X[['Crop_Type', 'Soil_Type']])
    X_encoded = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(['Crop_Type', 'Soil_Type']))

    # Standardization for numerical columns
    num_cols = X.select_dtypes(include=['number']).columns
    X_scaled = scaler.transform(X[num_cols])
    X_scaled = pd.DataFrame(X_scaled, columns=num_cols)

    # Combine transformed features
    X_transformed = pd.concat([X_scaled.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    return X_transformed
