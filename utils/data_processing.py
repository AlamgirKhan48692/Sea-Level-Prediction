import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

def load_and_prepare_data(file_path='epa-sea-level.csv'):
    """
    Load and prepare the EPA sea level data for analysis.
    
    Returns:
        pandas.DataFrame: Cleaned and prepared dataset
    """
    try:
        df = pd.read_csv(file_path)
        
        # Basic data cleaning
        df = df.dropna(subset=['CSIRO Adjusted Sea Level'])
        
        # Create additional features
        df['Decade'] = (df['Year'] // 10) * 10
        df['Years_Since_1880'] = df['Year'] - 1880
        df['Rate_of_Change'] = df['CSIRO Adjusted Sea Level'].diff()
        
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def get_basic_statistics(df):
    """
    Calculate basic statistics for the dataset.
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        dict: Dictionary containing basic statistics
    """
    stats = {
        'total_records': len(df),
        'year_range': f"{df['Year'].min()} - {df['Year'].max()}",
        'mean_sea_level': df['CSIRO Adjusted Sea Level'].mean(),
        'std_sea_level': df['CSIRO Adjusted Sea Level'].std(),
        'min_sea_level': df['CSIRO Adjusted Sea Level'].min(),
        'max_sea_level': df['CSIRO Adjusted Sea Level'].max(),
        'total_rise': df['CSIRO Adjusted Sea Level'].iloc[-1] - df['CSIRO Adjusted Sea Level'].iloc[0]
    }
    return stats

def prepare_ml_data(df, target_col='CSIRO Adjusted Sea Level', test_size=0.2):
    """
    Prepare data for machine learning models.
    
    Args:
        df (pandas.DataFrame): The dataset
        target_col (str): Target column name
        test_size (float): Test set size
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Feature selection
    feature_cols = ['Year', 'Years_Since_1880']
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test

def create_polynomial_features(X, degree=2):
    """
    Create polynomial features for the dataset.
    
    Args:
        X (array): Input features
        degree (int): Polynomial degree
        
    Returns:
        array: Polynomial features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly

def prepare_time_series_data(df, lookback=10):
    """
    Prepare data for LSTM time series prediction.
    
    Args:
        df (pandas.DataFrame): The dataset
        lookback (int): Number of previous time steps to use
        
    Returns:
        tuple: X, y arrays for LSTM
    """
    data = df['CSIRO Adjusted Sea Level'].values
    X, y = [], []
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)
