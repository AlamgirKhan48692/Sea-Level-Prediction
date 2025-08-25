# Sea Level Prediction Analysis

## Overview

This project is a comprehensive sea level prediction analysis application built with Streamlit. It analyzes EPA sea level data to create visualizations and machine learning models that predict future sea level changes. The application provides an interactive dashboard for exploring historical sea level trends, training multiple ML models (linear regression, polynomial regression, random forest, XGBoost, and LSTM), and generating future predictions with confidence intervals.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web application interface
- **Visualization Libraries**: 
  - Plotly for interactive charts and graphs
  - Matplotlib and Seaborn for static visualizations
- **Layout**: Wide layout with expandable sidebar for navigation and controls
- **Styling**: Custom CSS for notebook-style formatting with color-coded headers

### Backend Architecture
- **Data Processing**: Modular utility functions in `utils/data_processing.py`
  - Data loading and cleaning from CSV files
  - Feature engineering (decade grouping, rate of change calculations)
  - Train/test data preparation with sklearn preprocessing
- **Model Management**: Centralized model training and evaluation in `utils/models.py`
  - Multiple ML algorithms: Linear/Polynomial Regression, Random Forest, XGBoost, LSTM
  - Unified metrics calculation and model comparison
  - Future prediction capabilities with confidence intervals
- **Visualization Engine**: Specialized plotting functions in `utils/visualizations.py`
  - Interactive time series plots
  - Model prediction comparisons
  - Correlation analysis and statistical visualizations

### Data Storage Solutions
- **Primary Data Source**: CSV file (`epa-sea-level.csv`) containing EPA sea level measurements
- **Data Structure**: Time series data with year, CSIRO adjusted sea level, and error bounds
- **Feature Engineering**: Dynamic creation of derived features (decades, years since baseline, rate of change)

### Machine Learning Pipeline
- **Model Types**: 
  - Traditional ML: Linear/Polynomial Regression, Random Forest, XGBoost
  - Deep Learning: LSTM neural networks for time series forecasting
- **Evaluation Metrics**: MSE, R², MAE for comprehensive model assessment
- **Preprocessing**: StandardScaler and PolynomialFeatures for data normalization and feature expansion
- **Prediction Framework**: Future data generation with confidence intervals and uncertainty quantification

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework for dashboard interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning algorithms and preprocessing tools

### Visualization Dependencies
- **plotly**: Interactive plotting and dashboard components
- **matplotlib**: Static plotting and figure generation
- **seaborn**: Statistical data visualization

### Machine Learning Dependencies
- **xgboost**: Gradient boosting framework for advanced ML models
- **tensorflow/keras**: Deep learning framework for LSTM neural networks
- **scipy**: Scientific computing for statistical analysis and regression

### Data Processing
- **sklearn.preprocessing**: Data normalization and feature engineering
- **sklearn.model_selection**: Train/test splitting and cross-validation
- **sklearn.metrics**: Model evaluation and performance assessment

### Development and Testing
- **unittest**: Python testing framework for model validation
- **warnings**: Error handling and warning suppression for cleaner output