# Time Series Project: CO Pollution Analysis in Israel

## Overview
This project analyzes carbon monoxide (CO) pollution data in Israel over time, examining temporal patterns and relationships with vehicle registrations. The project employs various time series analysis techniques and forecasting models.

## Project Structure
- **data/**: Contains raw and processed datasets
  - `co_avg.csv`: Average CO values across stations
  - `co_half_monthly.csv`: Half-monthly CO measurements
  - `cars_transformed.csv`: Transformed vehicle registration data
  - `pollution_and_new_cars_unified.csv`: Combined pollution and vehicle data

- **pre_process/**: Data preprocessing scripts
  - `air_preprocessing.py`: Processes and transforms air quality data
  - `process_co_data.py`: Calculates average CO values across monitoring stations

- **visualizations/**: Data visualization outputs
  - Time series plots, forecasts, decompositions, and diagnostics
  - Statistical analysis visualizations

## Analysis Notebooks
- `01_pollution_eda.ipynb`: Initial exploratory data analysis
- `02_pollution_eda_final.ipynb`: Final exploratory analysis
- `03_models.ipynb`: Implementation of time series models
- `models.ipynb` & `models_2012.ipynb`: Additional modeling approaches
- `prophet_checks.ipynb`: Prophet model evaluation
- `change_point.ipynb`: Change point detection analysis
- `exo_comparison.ipynb`: Exogenous variable comparison

## Models Implemented
- SARIMA (Seasonal AutoRegressive Integrated Moving Average)
- Prophet (Facebook's forecasting tool)
- Kalman Filter models

## Key Features
- Time series decomposition and visualization
- Seasonal pattern analysis
- Change point detection
- Correlation analysis with exogenous variables
- Forecasting with various models
- Missing value impact analysis

## Statistical Analysis
The data shows these key characteristics:
- Mean CO Level: 0.4807
- Strong seasonality in pollution patterns
- Significant correlation with vehicle registrations
- Stationary time series (ADF p-value: 0.0004)

## Getting Started
1. Clone this repository
2. Ensure Python 3.x with required packages: pandas, numpy, matplotlib, statsmodels, prophet, etc.
3. Review notebooks in sequential order for full analysis flow

## Visualizations
The project includes various visualizations for:
- Seasonal decomposition
- Autocorrelation analysis
- Model diagnostics and forecasts
- Trend and seasonality components
