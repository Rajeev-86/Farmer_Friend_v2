import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

class Preprocessor:
    def __init__(self):
        self.encoding_maps = {}  # Store encoding maps for inference
        self.scaler = StandardScaler()  # Store scaler for standardization

    def fit(self, X_train, y_train):
        """Fit encoding maps and scaler."""
        categorical_features = ["Crop", "Region"]

        # Create encoding maps
        for feature in categorical_features:
            self.encoding_maps[feature] = y_train.groupby(X_train[feature]).mean()

        # Convert categorical features to numerical using the encoding maps before scaling
        X_train_encoded = X_train.copy()  # Create a copy of X_train to avoid modifying the original DataFrame
        for feature in categorical_features:
            X_train_encoded[feature] = X_train_encoded[feature].map(self.encoding_maps[feature])

        # Fit scaler on training features (now numerical)
        self.scaler.fit(X_train_encoded[categorical_features])  # Use the encoded features for scaling

    def transform(self, X):

        # Target encoding
        categorical_features = ["Crop", "Region"]
        for feature in categorical_features:
            X[feature] = X[feature].map(self.encoding_maps.get(feature, {})).fillna(self.encoding_maps[feature].mean())

        # Handle date column
        if "Year" in X.columns and "Month" in X.columns:
            X['Date'] = pd.to_datetime(X[['Year', 'Month']].assign(DAY=1))
            X.set_index('Date', inplace=True)
            X.drop(columns=['Year', 'Month'], inplace=True)

            # Apply reindexing (for consistency in time-series data)
            X.index = pd.date_range(start=X.index.min(), periods=len(X), freq=pd.infer_freq(X.index))

            X = X.reset_index()

        # Standardization
        X[categorical_features] = self.scaler.transform(X[categorical_features])
        X = X.apply(pd.to_numeric, errors='coerce')
        return X
