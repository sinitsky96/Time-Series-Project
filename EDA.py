import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, stats
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import os

def ensure_visualization_dir():
    # Create visualizations directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

def load_data():
    df = pd.read_csv('data/air_data_transformed.csv')
    df['תאריך ושעה'] = pd.to_datetime(df['תאריך ושעה'])
    return df.sort_values('תאריך ושעה')

def create_fft_heatmap(df):
    values = df['CO_Israel'].values
    window_size = 24
    step_size = 6
    
    windows = []
    times = []
    
    for i in range(0, len(values) - window_size, step_size):
        window = values[i:i + window_size]
        windows.append(np.abs(fft.fft(window))[1:window_size//2])
        times.append(df['תאריך ושעה'].iloc[i + window_size//2])
    
    plt.figure(figsize=(12, 6))
    plt.imshow(np.array(windows).T, aspect='auto', cmap='hot',
              extent=[pd.Timestamp(times[0]).year, pd.Timestamp(times[-1]).year, 0, 12])
    
    plt.colorbar(label='Magnitude')
    plt.ylabel('Frequency (cycles per year)')
    plt.xlabel('Year')
    plt.title('FFT Heatmap of CO Levels in Israel')
    plt.savefig('visualizations/fft_heatmap.png')
    plt.close()

def seasonal_analysis(df):
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(df['CO_Israel'], period=24, model='additive')
    
    plt.figure(figsize=(15, 10))
    plt.subplot(411)
    plt.plot(df['תאריך ושעה'], df['CO_Israel'])
    plt.title('Original Time Series')
    
    plt.subplot(412)
    plt.plot(df['תאריך ושעה'], decomposition.trend)
    plt.title('Trend')
    
    plt.subplot(413)
    plt.plot(df['תאריך ושעה'], decomposition.seasonal)
    plt.title('Seasonal')
    
    plt.subplot(414)
    plt.plot(df['תאריך ושעה'], decomposition.resid)
    plt.title('Residual')
    
    plt.tight_layout()
    plt.savefig('visualizations/seasonal_decomposition.png')
    plt.close()

def yearly_boxplot(df):
    # Add year column
    df['Year'] = df['תאריך ושעה'].dt.year
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df, x='Year', y='CO_Israel')
    plt.title('Yearly Distribution of CO Levels')
    plt.xticks(rotation=45)
    plt.savefig('visualizations/yearly_boxplot.png')
    plt.close()

def basic_statistics(df):
    # Calculate comprehensive statistics for model selection
    stats_dict = {
        'Mean': df['CO_Israel'].mean(),
        'Median': df['CO_Israel'].median(),
        'Std Dev': df['CO_Israel'].std(),
        'Variance': df['CO_Israel'].var(),
        'Skewness': df['CO_Israel'].skew(),
        'Kurtosis': df['CO_Israel'].kurtosis(),
        'Min': df['CO_Israel'].min(),
        'Max': df['CO_Israel'].max(),
        '25th Percentile': df['CO_Israel'].quantile(0.25),
        '75th Percentile': df['CO_Israel'].quantile(0.75),
        'Count': df['CO_Israel'].count(),
        'Missing Values': df['CO_Israel'].isnull().sum()
    }
    
    # Coefficient of Variation (if mean is not zero)
    if df['CO_Israel'].mean() != 0:
        stats_dict['Coefficient of Variation'] = df['CO_Israel'].std() / df['CO_Israel'].mean()
    else:
        stats_dict['Coefficient of Variation'] = np.nan

    # Stationarity Test: Augmented Dickey-Fuller test
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(df['CO_Israel'].dropna())
    stats_dict['ADF Statistic'] = adf_result[0]
    stats_dict['ADF p-value'] = adf_result[1]
    stats_dict['ADF # Lags'] = adf_result[2]
    stats_dict['ADF # Observations'] = adf_result[3]
    stats_dict['ADF Critical Values'] = adf_result[4]

    # Stationarity Test: KPSS test with warning suppression for InterpolationWarning
    try:
        import warnings
        from statsmodels.tsa.stattools import kpss
        from statsmodels.tools.sm_exceptions import InterpolationWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            kpss_result = kpss(df['CO_Israel'].dropna(), regression='c', nlags="auto")
            
        stats_dict['KPSS Statistic'] = kpss_result[0]
        stats_dict['KPSS p-value'] = kpss_result[1]
        stats_dict['KPSS # Lags'] = kpss_result[2]
        stats_dict['KPSS Critical Values'] = kpss_result[3]
    except Exception as e:
        stats_dict['KPSS Test'] = f"Error performing KPSS test: {e}"

    # Save statistics to file
    with open('visualizations/statistics_summary.txt', 'w') as f:
        f.write("Comprehensive Statistics for CO Levels in Israel\n")
        f.write("=================================================\n\n")
        for stat, value in stats_dict.items():
            if stat in ['ADF Critical Values', 'KPSS Critical Values'] and isinstance(value, dict):
                f.write(f"{stat}:\n")
                for key, val in value.items():
                    f.write(f"    {key}: {val:.4f}\n")
            else:
                try:
                    f.write(f"{stat}: {value:.4f}\n")
                except Exception:
                    f.write(f"{stat}: {value}\n")

def additional_analysis(df):
    # 1. Autocorrelation analysis (using Yearly Integration)
    plt.figure(figsize=(10, 4))
    # Use Yearly_Integration if available; drop missing values due to the shift operation.
    if 'Yearly_Integration' in df.columns:
        plot_acf(df['Yearly_Integration'].dropna(), lags=50)
        plt.title('Autocorrelation of Yearly Integration of CO Levels')
    else:
        plot_acf(df['CO_Israel'], lags=50)
        plt.title('Autocorrelation of CO Levels')
    plt.savefig('visualizations/autocorrelation.png')
    plt.close()
    
    # 2. Rolling statistics
    window = 24  # 1 year
    rolling_mean = df['CO_Israel'].rolling(window=window).mean()
    rolling_std = df['CO_Israel'].rolling(window=window).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['תאריך ושעה'], rolling_mean, label='Rolling Mean')
    plt.plot(df['תאריך ושעה'], rolling_std, label='Rolling Std')
    plt.legend()
    plt.title('Rolling Statistics (1-year window)')
    plt.savefig('visualizations/rolling_stats.png')
    plt.close()
    
    # 3. Year-over-year change
    df['YoY_Change'] = df['CO_Israel'].pct_change(periods=24)  # 24 periods = 1 year
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['תאריך ושעה'], df['YoY_Change'])
    plt.title('Year-over-Year Change in CO Levels')
    plt.savefig('visualizations/yoy_change.png')
    plt.close()

def yearly_integration(df):
    # Calculate difference between each point and its value 12 months later
    # Since we have 2 samples per month, we use 24 periods
    df['Yearly_Integration'] = df['CO_Israel'] - df['CO_Israel'].shift(-24)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the integration
    plt.subplot(211)
    plt.plot(df['תאריך ושעה'], df['Yearly_Integration'], 'b-', label='Yearly Integration')
    plt.title('Yearly Integration (y_k - y_{k+12})')
    plt.ylabel('Difference in CO Levels')
    plt.legend()
    
    # Plot histogram of the integration values
    plt.subplot(212)
    plt.hist(df['Yearly_Integration'].dropna(), bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Yearly Integration Values')
    plt.xlabel('Difference in CO Levels')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('visualizations/yearly_integration.png')
    plt.close()
    
    # Calculate some statistics about the integration
    integration_stats = {
        'Mean Difference': df['Yearly_Integration'].mean(),
        'Std Dev of Difference': df['Yearly_Integration'].std(),
        'Max Increase': df['Yearly_Integration'].max(),
        'Max Decrease': df['Yearly_Integration'].min(),
    }
    
    # Append to statistics file
    with open('visualizations/statistics_summary.txt', 'a') as f:
        f.write("\n\nYearly Integration Statistics\n")
        f.write("===========================\n")
        for stat, value in integration_stats.items():
            f.write(f"{stat}: {value:.4f}\n")

def run_analysis():
    ensure_visualization_dir()
    df = load_data()
    
    # Run all analyses
    create_fft_heatmap(df)
    seasonal_analysis(df)
    yearly_boxplot(df)
    basic_statistics(df)
    yearly_integration(df)
    additional_analysis(df)
    
    print("Analysis completed. The following files have been created in the 'visualizations' folder:")
    print("1. fft_heatmap.png - Frequency analysis over time")
    print("2. seasonal_decomposition.png - Trend, seasonal, and residual components")
    print("3. yearly_boxplot.png - Yearly distribution of CO levels")
    print("4. statistics_summary.txt - Basic statistical measures")
    print("5. autocorrelation.png - Autocorrelation plot (of Yearly Integration)")
    print("6. rolling_stats.png - Rolling statistics")
    print("7. yoy_change.png - Year-over-year change")
    print("8. yearly_integration.png - Yearly integration analysis")

if __name__ == "__main__":
    run_analysis()
