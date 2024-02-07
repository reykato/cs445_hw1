import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np

# Read the DataFrame from the CSV file


def clean_dataframe(df_path, nominal_columns, ordinal_columns, ordinal_mapping):
    """
    Use data cleaning and transformation techinques to clean data for processing
    """
    df = pd.read_csv(df_path)


    # ----- PREPROCESSING FOR ORDINAL COLUMNS -----\
    # Remove whitespace
    for column in ordinal_columns:
        df[column] = df[column].str.strip()

    # Normalize cases
    for column in ordinal_columns:
        df[column] = df[column].str.upper()

    # Map Feature D into binary values
    for column in ordinal_columns:
        df[column] = df[column].map(ordinal_mapping)


    # ----- PREPROCESSING FOR NOMINAL COLUMNS -----
    # Check for NaN values in specified columns and replace them with the mean value of the column
    for column in nominal_columns:
        df[column].fillna(df[column].mean(), inplace=True)

    # Remove outliers using z-score
    z_scores = zscore(df[nominal_cols])
    filtered_entries = (abs(z_scores) < 2).all(axis=1)
    df = df[filtered_entries]

    # Perform min-max scaling on specified columns
    for column in nominal_columns:
        min_val = df[column].min()
        max_val = df[column].max()
        df[column] = (df[column] - min_val) / (max_val - min_val)

    return df

def display_dataframe(df: pd.DataFrame):
    """
    Use matplotlib to display a pie chart
    """
    vals = [df[x].mean() for x in df.columns if x != 'ID']
    labels = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E', 'Feature F', 'Feature G', 'Feature H']

    plt.legend(loc='best')
    plt.pie(vals, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.show() 


def evaluate_correlation_with_target(df):
    """
    Evaluate Pearson correlation between the features and the target variable.
    """
 
    return df.corr()['quality'].drop('quality')

def rank_features(correlation_with_target):
    """
    Rank the features based on their correlation with the target variable (quality).
    """
    return correlation_with_target.abs().sort_values(ascending=False)

def evaluate_correlation_matrix(df):
    """
    Evaluate and show the correlation matrix for all features.
    """
    return df.corr()

def find_redundant_features(correlation_matrix, threshold=0.65):
    """
    Find redundant features based on the correlation matrix with the given threshold.
    """
    redundant_features = []
    for i, column_i in enumerate(correlation_matrix.columns):
        for j, column_j in enumerate(correlation_matrix.columns[:i]):
            if abs(correlation_matrix.iloc[i, j]) >= threshold:
                redundant_features.append((column_i, column_j))
    return redundant_features



ordinal_map = {'ABNORMAL':0, 'NORMAL':1, 'A':0, 'B':0.5, 'C':1}
nominal_cols = ['Feature A', 'Feature B', 'Feature E', 'Feature H', 'Feature G']
ordinal_cols = ['Feature C', 'Feature D']
cleaned_df = clean_dataframe('assign1_data1.csv', nominal_cols, ordinal_cols, ordinal_map)
print(cleaned_df)
# display_dataframe(cleaned_df)

df = pd.read_csv("assign1_data2.csv")

correlation_with_target = evaluate_correlation_with_target(df)
print("Pearson correlation between features and target variable:")
print(correlation_with_target)

ranked_features = rank_features(correlation_with_target)
print("\nRanked features based on correlation with target variable:")
print(ranked_features)

correlation_matrix = evaluate_correlation_matrix(df)
print("\nCorrelation matrix for all features:")
print(correlation_matrix)

redundant_features = find_redundant_features(correlation_matrix, threshold=0.65)
print("\nRedundant features (correlation >= 0.65):")
for feature_pair in redundant_features:
    print(feature_pair)