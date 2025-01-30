import pandas as pd
import numpy as np

def load_and_transform_data():
    # Read the CSV file
    # Skip the first 2 rows (metadata) and use the 3rd row as header
    df = pd.read_csv('air_data.csv', skiprows=2, header=0)
    
    # Replace '<Samp' values with 0
    df = df.replace('<Samp', 0)
    
    # Convert all non-numeric values (except dates) to NaN
    numeric_columns = df.columns[1:]  # All columns except the date column
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Convert negative values to 0
    df[numeric_columns] = df[numeric_columns].clip(lower=0)
    
    # Calculate row means excluding NaN values
    # axis=1 means calculate across columns
    # skipna=True means ignore NaN values
    df['CO_Israel'] = df[numeric_columns].mean(axis=1, skipna=True)
    
    # Convert 24:00 to 00:00 and adjust the date
    df['תאריך ושעה'] = df['תאריך ושעה'].str.replace('24:00', '00:00')
    # Convert date column to datetime
    df['תאריך ושעה'] = pd.to_datetime(df['תאריך ושעה'], format='%H:%M %d/%m/%Y')
    # Add one day to adjust for the 24:00 -> 00:00 conversion
    df['תאריך ושעה'] = df['תאריך ושעה'] + pd.Timedelta(days=1)
    
    # Create a column for first/second half of month (1-15 = 1, 16-31 = 2)
    df['half_month'] = df['תאריך ושעה'].dt.day.apply(lambda x: 1 if x <= 15 else 2)
    
    # Group by year, month and half_month to get two samples per month
    df_downsampled = df.groupby([
        df['תאריך ושעה'].dt.year.rename('year'),
        df['תאריך ושעה'].dt.month.rename('month'),
        'half_month'
    ])['CO_Israel'].mean().reset_index()
    
    # Create proper datetime for the downsampled data (using the 1st and 16th of each month)
    df_downsampled['תאריך ושעה'] = pd.to_datetime(
        df_downsampled.apply(
            lambda x: f"{int(x['year'])}-{int(x['month'])}-{1 if x['half_month']==1 else 16}",
            axis=1
        )
    )
    
    # Keep only the date and CO_Israel columns
    df_downsampled = df_downsampled[['תאריך ושעה', 'CO_Israel']]
    
    return df_downsampled

if __name__ == "__main__":
    # Load and transform the data
    transformed_data = load_and_transform_data()
    
    # Save the transformed data to a new CSV file
    transformed_data.to_csv('air_data_transformed.csv', index=False)
    
    print("Data transformation completed. Saved to 'air_data_transformed.csv'")
