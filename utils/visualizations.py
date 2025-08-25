import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_time_series(df, title="Sea Level Rise Over Time"):
    """
    Create an interactive time series plot using Plotly.
    
    Args:
        df (pandas.DataFrame): Dataset with Year and sea level data
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    fig = go.Figure()
    
    # Add scatter plot for actual data
    fig.add_trace(go.Scatter(
        x=df['Year'],
        y=df['CSIRO Adjusted Sea Level'],
        mode='markers',
        name='CSIRO Adjusted Sea Level',
        marker=dict(size=6, color='blue', opacity=0.7),
        hovertemplate='Year: %{x}<br>Sea Level: %{y:.2f} inches<extra></extra>'
    ))
    
    # Add error bounds if available
    if 'Lower Error Bound' in df.columns and 'Upper Error Bound' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Upper Error Bound'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False,
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Lower Error Bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='Error Bounds',
            fillcolor='rgba(0,100,80,0.2)'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Sea Level (inches)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_decade_analysis(df):
    """
    Create decade-wise analysis plots.
    
    Args:
        df (pandas.DataFrame): Dataset with decade information
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Decade averages
    decade_avg = df.groupby('Decade')['CSIRO Adjusted Sea Level'].mean()
    axes[0, 0].bar(decade_avg.index, decade_avg.values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Average Sea Level by Decade')
    axes[0, 0].set_xlabel('Decade')
    axes[0, 0].set_ylabel('Average Sea Level (inches)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Rate of change by decade
    decade_change = df.groupby('Decade')['Rate_of_Change'].mean()
    axes[0, 1].plot(decade_change.index, decade_change.values, marker='o', color='red', linewidth=2)
    axes[0, 1].set_title('Average Rate of Change by Decade')
    axes[0, 1].set_xlabel('Decade')
    axes[0, 1].set_ylabel('Rate of Change (inches/year)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Distribution of sea levels
    axes[1, 0].hist(df['CSIRO Adjusted Sea Level'], bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Sea Level Measurements')
    axes[1, 0].set_xlabel('Sea Level (inches)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Cumulative sea level rise
    cumulative_rise = df['CSIRO Adjusted Sea Level'] - df['CSIRO Adjusted Sea Level'].iloc[0]
    axes[1, 1].plot(df['Year'], cumulative_rise, color='purple', linewidth=2)
    axes[1, 1].set_title('Cumulative Sea Level Rise Since 1880')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Cumulative Rise (inches)')
    
    plt.tight_layout()
    return fig

def plot_model_predictions(df, models, X_test, y_test, test_years):
    """
    Plot predictions from different models.
    
    Args:
        df (pandas.DataFrame): Original dataset
        models (dict): Dictionary of trained models
        X_test, y_test: Test data
        test_years: Years corresponding to test data
        
    Returns:
        plotly.graph_objects.Figure: Interactive comparison plot
    """
    fig = go.Figure()
    
    # Add original data
    fig.add_trace(go.Scatter(
        x=df['Year'],
        y=df['CSIRO Adjusted Sea Level'],
        mode='markers',
        name='Original Data',
        marker=dict(size=4, color='blue', opacity=0.6)
    ))
    
    # Add test data
    fig.add_trace(go.Scatter(
        x=test_years,
        y=y_test,
        mode='markers',
        name='Test Data',
        marker=dict(size=6, color='red', symbol='x')
    ))
    
    # Add model predictions
    colors = ['green', 'orange', 'purple', 'brown', 'pink']
    for i, (model_name, predictions) in enumerate(models.predictions.items()):
        fig.add_trace(go.Scatter(
            x=test_years,
            y=predictions,
            mode='lines',
            name=f'{model_name} Prediction',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Year',
        yaxis_title='Sea Level (inches)',
        hovermode='x unified',
        height=600
    )
    
    return fig

def plot_future_predictions(df, future_predictions, future_years, confidence_intervals=None):
    """
    Plot future predictions with confidence intervals.
    
    Args:
        df (pandas.DataFrame): Historical data
        future_predictions (dict): Dictionary of future predictions by model
        future_years (array): Future years
        confidence_intervals (dict): Confidence intervals for predictions
        
    Returns:
        plotly.graph_objects.Figure: Future predictions plot
    """
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df['Year'],
        y=df['CSIRO Adjusted Sea Level'],
        mode='markers',
        name='Historical Data',
        marker=dict(size=4, color='blue', opacity=0.6)
    ))
    
    # Add future predictions
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, predictions) in enumerate(future_predictions.items()):
        fig.add_trace(go.Scatter(
            x=future_years,
            y=predictions,
            mode='lines',
            name=f'{model_name} Future',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
        ))
        
        # Add confidence intervals if available
        if confidence_intervals and model_name in confidence_intervals:
            upper_bound = predictions + confidence_intervals[model_name]
            lower_bound = predictions - confidence_intervals[model_name]
            
            fig.add_trace(go.Scatter(
                x=future_years,
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0)',
                showlegend=False,
                name=f'{model_name} Upper'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_years,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0)',
                name=f'{model_name} Confidence',
                fillcolor=f'rgba({colors[i % len(colors)]},0.2)'
            ))
    
    fig.update_layout(
        title='Future Sea Level Predictions (2014-2050)',
        xaxis_title='Year',
        yaxis_title='Sea Level (inches)',
        hovermode='x unified',
        height=600
    )
    
    return fig

def plot_metrics_comparison(metrics_df):
    """
    Create bar plots comparing model metrics.
    
    Args:
        metrics_df (pandas.DataFrame): DataFrame with model metrics
        
    Returns:
        plotly.graph_objects.Figure: Metrics comparison plot
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Root Mean Square Error', 'Mean Absolute Error', 'R² Score', 'Mean Square Error'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    models = metrics_df.index
    colors = px.colors.qualitative.Set3
    
    # RMSE
    fig.add_trace(
        go.Bar(x=models, y=metrics_df['RMSE'], name='RMSE', marker_color=colors[0]),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(x=models, y=metrics_df['MAE'], name='MAE', marker_color=colors[1]),
        row=1, col=2
    )
    
    # R²
    fig.add_trace(
        go.Bar(x=models, y=metrics_df['R²'], name='R²', marker_color=colors[2]),
        row=2, col=1
    )
    
    # MSE
    fig.add_trace(
        go.Bar(x=models, y=metrics_df['MSE'], name='MSE', marker_color=colors[3]),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Model Performance Metrics Comparison")
    
    return fig

def create_correlation_heatmap(df):
    """
    Create correlation heatmap for numerical variables.
    
    Args:
        df (pandas.DataFrame): Dataset
        
    Returns:
        matplotlib.figure.Figure: Correlation heatmap
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix of Numerical Variables')
    
    return fig
