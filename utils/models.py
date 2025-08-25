import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class SeaLevelModels:
    """
    Collection of machine learning models for sea level prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train linear regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        self.models['Linear Regression'] = model
        self.predictions['Linear Regression'] = y_pred
        self.metrics['Linear Regression'] = self._calculate_metrics(y_test, y_pred)
        
        return model, y_pred
    
    def train_polynomial_regression(self, X_train, y_train, X_test, y_test, degree=2):
        """Train polynomial regression model."""
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        y_pred = model.predict(X_test_poly)
        
        self.models['Polynomial Regression'] = {'model': model, 'poly': poly}
        self.predictions['Polynomial Regression'] = y_pred
        self.metrics['Polynomial Regression'] = self._calculate_metrics(y_test, y_pred)
        
        return model, y_pred
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """Train Random Forest model."""
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        self.models['Random Forest'] = model
        self.predictions['Random Forest'] = y_pred
        self.metrics['Random Forest'] = self._calculate_metrics(y_test, y_pred)
        
        return model, y_pred
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model."""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        self.models['XGBoost'] = model
        self.predictions['XGBoost'] = y_pred
        self.metrics['XGBoost'] = self._calculate_metrics(y_test, y_pred)
        
        return model, y_pred
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train LSTM model."""
        # Reshape data for LSTM
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train the model
        history = model.fit(
            X_train_lstm, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )
        
        y_pred = model.predict(X_test_lstm).flatten()
        
        self.models['LSTM'] = model
        self.predictions['LSTM'] = y_pred
        self.metrics['LSTM'] = self._calculate_metrics(y_test, y_pred)
        
        return model, y_pred, history
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
    
    def get_metrics_comparison(self):
        """Get comparison of all model metrics."""
        if not self.metrics:
            return None
        
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.round(4)
        return comparison_df
    
    def predict_future(self, model_name, X_future, scaler=None):
        """Make future predictions with a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if model_name == 'Polynomial Regression':
            X_future_poly = model['poly'].transform(X_future)
            predictions = model['model'].predict(X_future_poly)
        elif model_name == 'LSTM':
            X_future_lstm = X_future.reshape((X_future.shape[0], X_future.shape[1], 1))
            predictions = model.predict(X_future_lstm).flatten()
        else:
            predictions = model.predict(X_future)
        
        return predictions

def create_future_data(last_year, years_ahead=30):
    """Create future data for predictions."""
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
    future_data = pd.DataFrame({
        'Year': future_years,
        'Years_Since_1880': future_years - 1880
    })
    return future_data
