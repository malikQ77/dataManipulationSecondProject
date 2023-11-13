import pandas as pd
import numpy as np

def filter_cols(df, missing_threshold=0.9):
    """
    Filters out columns from a DataFrame based on:
    1. Missing values exceeding the given threshold, or
    2. "Unnamed" prefixes.
    
    The function provides a structured report on the columns 
    that are filtered out and the respective reasons as a DataFrame.
    
    Parameters:
    - df: Input pandas DataFrame
    - missing_threshold: Proportion threshold for missing values (default is 0.9)
    
    Returns:
    - Tuple: Filtered DataFrame, Report DataFrame with concatenated reasons
    """
    
    threshold_count = missing_threshold * len(df)
    
    # Dictionary to store columns and their removal reasons
    cols_to_filter = {}
    
    for col in df.columns:
        reasons = []
        if df[col].isnull().sum() > threshold_count:
            reasons.append('Excessive Missing Values')
        if col.lower().startswith('unnamed'):
            reasons.append('Unnamed Column')
        
        if reasons:
            cols_to_filter[col] = ', '.join(reasons)
    
    # Generate the filtered dataframe
    df_filtered = df.drop(columns=cols_to_filter.keys())
    
    # Prepare the report data
    report_data = {
        'Column': list(cols_to_filter.keys()),
        'Reason': list(cols_to_filter.values())
    }
    
    # Convert report data to a DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Add a title to the DataFrame by adding a row at the top
    title_df = pd.DataFrame([['Filtered Columns Report', '']], columns=['Column', 'Reason'])
    report_with_title = pd.concat([title_df, report_df], ignore_index=True)
    
    return df_filtered, report_df

# ----------------------------------------------------------------------------------------

def format_number(x):
    """
    If x is a number and not NaN, format it to two decimal places or as an integer if it's whole.
    If x is NaN, return it unchanged.
    """
    if pd.isnull(x):  # Check for NaN
        return x
    if isinstance(x, float):
        return round(x, 2)
    return x


# ----------------------------------------------------------------------------------------

def descriptive_analysis(df):
    """
    Perform a descriptive analysis on a pandas DataFrame, handling both numeric and non-numeric data.
    Formats numeric data with two decimal places or as integers and adds headers for numeric and non-numeric data.
    """
    # Prepare the statistics for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.any():
        numeric_stats = df[numeric_cols].describe().applymap(format_number)
        numeric_stats.loc['missing'] = df[numeric_cols].isnull().sum().apply(format_number)
        numeric_stats.loc['freq'] = df[numeric_cols].apply(lambda x: x.dropna().value_counts().iloc[0] if not x.dropna().empty else np.nan).apply(format_number)
        numeric_stats.columns = pd.MultiIndex.from_product([['Numeric Data'], numeric_stats.columns])  # Add a header for numeric data

    # Prepare the statistics for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns
    if non_numeric_cols.any():
        non_numeric_stats = df[non_numeric_cols].describe(include='all')
        non_numeric_stats.loc['missing'] = df[non_numeric_cols].isnull().sum()
        non_numeric_stats.loc['freq'] = df[non_numeric_cols].apply(lambda x: x.value_counts().iloc[0] if not x.empty else np.nan)
        non_numeric_stats.columns = pd.MultiIndex.from_product([['Non-Numeric Data'], non_numeric_stats.columns])  # Add a header for non-numeric data

    # Combine the statistics
    combined_stats = pd.concat([numeric_stats, non_numeric_stats], axis=1)
    return combined_stats.T


if __name__ == "__main__":
    print("getting global_functions script successfully")
