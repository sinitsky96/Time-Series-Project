import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.graphics import tsaplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from prophet.plot import plot_components
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from statsmodels.graphics import tsaplots
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from sklearn.preprocessing import MinMaxScaler
# Suppress specific statsmodels warnings about frequency
warnings.filterwarnings('ignore', 'No frequency information was provided')
warnings.filterwarnings('ignore', 'No supported index is available')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# Load the datasets
df_air = pd.read_csv('data/co_half_monthly.csv', parse_dates=['Date'])
df_cars = pd.read_csv('data/cars_transformed.csv', parse_dates=['Date'])

# Set the Date column as the index for cars data
df_cars.set_index('Date', inplace=True)
dfCars = df_cars[['car_num']]

# Filter data from 2012 to end of 2024
dfCars = dfCars[(dfCars.index >= '2012-01-01') & (dfCars.index <= '2024-12-31')]
yearly_integration_cars = dfCars.diff(periods=24).dropna()

def find_max_correlation(series1, series2, max_lag=48):
    """Find maximum correlation and optimal lag"""
    lag_range = range(-max_lag, 1)
    correlations = []
    for lag in lag_range:
        if lag == 0:
            corr = series1.corr(series2)
        else:
            corr = series1.shift(lag).corr(series2)
        correlations.append((lag, corr, abs(corr)))
    
    corr_df = pd.DataFrame(correlations, columns=['lag', 'correlation', 'abs_correlation'])
    optimal_lag = corr_df.loc[corr_df['correlation'].idxmax()]
    return optimal_lag, corr_df

def process_station_data(station_data, yearly_integration_cars):
    # Filter data from 2012 to end of 2024
    station_data = station_data[(station_data.index >= '2012-01-01') & (station_data.index <= '2024-12-31')]
    yearly_integration_co = station_data.diff(periods=24).dropna()
    
    # Ensure both series have the same index
    common_index = yearly_integration_co.index.intersection(yearly_integration_cars.index)
    yearly_integration_co = yearly_integration_co.loc[common_index]
    yearly_integration_cars = yearly_integration_cars.loc[common_index]
    
    # Normalize the data
    scaler = MinMaxScaler()
    yearly_integration_co_scaled = scaler.fit_transform(yearly_integration_co)
    yearly_integration_cars_scaled = scaler.fit_transform(yearly_integration_cars)
    
    # Create DataFrame with both series
    combined_df = pd.DataFrame({
        'co': yearly_integration_co_scaled.flatten(),
        'cars': yearly_integration_cars_scaled.flatten()
    }, index=common_index)
    
    # Apply rolling average
    window = 6  # 3 months of bi-monthly data
    roll_co = combined_df['co'].rolling(window=window, center=True).mean().dropna()
    roll_cars = combined_df['cars'].rolling(window=window, center=True).mean().dropna()
    
    return roll_co, roll_cars

# Get unique stations
stations = df_air['station'].unique()

# Create subplots in a grid
n_stations = len(stations)
n_cols = 2
n_rows = (n_stations + 1) // 2  # Round up division

# First figure: Correlation Analysis
plt.figure(figsize=(15, 5*n_rows))

for idx, station in enumerate(stations, 1):
    # Filter data for current station
    station_data = df_air[df_air['station'] == station].set_index('Date')[['CO']]
    roll_co, roll_cars = process_station_data(station_data, yearly_integration_cars)
    
    # Calculate optimal lag
    optimal_lag, corr_df = find_max_correlation(roll_co, roll_cars)
    
    # Create subplot
    plt.subplot(n_rows, n_cols, idx)
    
    # Plot correlation analysis
    plt.plot(corr_df['lag'], corr_df['correlation'], marker='o')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title(f'{station}\nMax Correlation: {optimal_lag["correlation"]:.4f} at lag {optimal_lag["lag"]}')
    plt.xlabel('Lag (positive = CO leading, negative = CO lagging)')
    plt.ylabel('Correlation')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Second figure: Time Series Comparison
plt.figure(figsize=(15, 5*n_rows))

for idx, station in enumerate(stations, 1):
    # Filter data for current station
    station_data = df_air[df_air['station'] == station].set_index('Date')[['CO']]
    roll_co, roll_cars = process_station_data(station_data, yearly_integration_cars)
    
    # Calculate optimal lag
    optimal_lag, _ = find_max_correlation(roll_co, roll_cars)
    
    # Create subplot
    plt.subplot(n_rows, n_cols, idx)
    
    # Plot time series with optimal lag
    shifted_co = roll_co.shift(int(optimal_lag['lag']))
    plt.plot(shifted_co.index, shifted_co, label='CO (shifted)', linewidth=2)
    plt.plot(roll_cars.index, roll_cars, label='Cars', linewidth=2)
    plt.title(f'{station}\nCorrelation: {optimal_lag["correlation"]:.4f}\nShift: {optimal_lag["lag"]/2:.1f} months')
    plt.xlabel('Date')
    plt.ylabel('Normalized Rolling Average')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary for all stations
print("\nRolling Average Analysis Summary for All Stations:")
print(f"Window size: 6 periods (3 months)")
print("\nStation-wise Results:")
print("-" * 80)
for station in stations:
    station_data = df_air[df_air['station'] == station].set_index('Date')[['CO']]
    roll_co, roll_cars = process_station_data(station_data, yearly_integration_cars)
    optimal_lag, _ = find_max_correlation(roll_co, roll_cars)
    print(f"\nStation: {station}")
    print(f"Optimal lag: {optimal_lag['lag']} periods ({optimal_lag['lag']/2:.1f} months)")
    print(f"Maximum correlation: {optimal_lag['correlation']:.4f}")