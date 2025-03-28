import pandas as pd
import numpy as np

def process_co_data():
    """
    Reads CO data from csv file, calculates the average CO value per date across all stations,
    excluding stations with missing values for each date, and saves the result to a new file.
    
    The script:
    1. Reads the CO data from co_clean.csv
    2. Removes rows with missing CO values
    3. Groups data by date and calculates the average CO value across all stations
    4. Saves the results to a new CSV file
    """
    # Read the CO data
    print("Reading CO data from file...")
    df = pd.read_csv(r'D:\Study Docs\Degree Material\Sem 9 proj\TS\Time-Series-Project\data\co_half_monthly.csv')
    
    # Convert date_time to datetime format for proper date handling
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove rows with empty CO values
    # This ensures that stations without values for a particular date 
    # are not included in the calculation for that date
    df = df[df['CO'].notna()]
    
    # Group by date and calculate average CO value across all stations
    print("Calculating average CO values per date...")
    avg_co = df.groupby('date')['CO'].mean().reset_index()
    
    # Count number of stations with data for each date (for informational purposes)
    station_counts = df.groupby('date')['station'].nunique().reset_index()
    station_counts.columns = ['date', 'station_count']
    
    # Merge the average CO values with station counts
    avg_co = pd.merge(avg_co, station_counts, on='date')
    
    # Rename columns for clarity
    avg_co.columns = ['Date', 'CO', 'StationCount']
    
    # Save to new CSV file
    output_file = 'data/co_avg.csv'
    print(f"Saving results to {output_file}...")
    avg_co.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"\nProcessing complete. Data saved to {output_file}")
    print(f"Number of dates processed: {len(avg_co)}")
    print(f"Date range: {avg_co['Date'].min()} to {avg_co['Date'].max()}")
    print(f"Average CO value across all dates: {avg_co['CO'].mean():.4f}")
    print(f"Average number of stations per date: {avg_co['StationCount'].mean():.1f}")
    print(f"Maximum stations for any date: {avg_co['StationCount'].max()}")
    
    return avg_co

if __name__ == "__main__":
    process_co_data() 