import pandas as pd

def clean_dataframe(df_path, columns_to_clean):
    # Read the DataFrame from the CSV file
    df = pd.read_csv(df_path)
    
    # Check for NaN values in specified columns and replace them with the mean value of the column
    for column in columns_to_clean:
        if column in df.columns:
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            print(f"Column '{column}' not found in DataFrame.")
    
    return df

# Usage
columns_to_clean = ['Feature A', 'Feature B', 'Feature E', 'Feature H', 'Feature G']
cleaned_df = clean_dataframe('assign1_data1.csv', columns_to_clean)
print(cleaned_df)
