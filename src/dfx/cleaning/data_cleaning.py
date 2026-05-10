import pandas as pd
import numpy as np


def suggest_fill_strategy(df, skew_threshold=1.0, rare_category_threshold=0.05, high_missing_threshold=50.0,high_missing_action="drop_column"):
    """
    Analyzes a DataFrame and suggests intelligent strategies for handling missing values in each column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to analyze.
    skew_threshold : float, default=1.0
        Absolute skewness value above which a column is considered skewed.
    rare_category_threshold : float, default=0.05
        Frequency threshold below which a category is considered 'rare'.
    high_missing_threshold : float, default=50.0
        Missing percentage above which dropping the column is suggested.

    Returns:
    --------
    pandas.DataFrame
        A summary with suggestions for each column containing missing values.
    """
    
    suggestions = {}
    
    for col in df.columns:
        # Calculate missing stats
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        
        # Skip columns with no missing values
        if null_count == 0:
            continue
        
        dtype = df[col].dtype
        col_data = df[col].dropna()
        info = {
            "column_name": col,
            "dtype": dtype,
            "missing_count": null_count,
            "missing_percent": round(null_pct, 2),
            "suggested_strategy": None,
            "rationale": None,
            "warning": None
        }
        
        # 1. Check for high missing percentage
        if null_pct > high_missing_threshold:
            info["suggested_strategy"] = high_missing_action
            info["rationale"] = f"Over {high_missing_threshold}% of data is missing"
            info["warning"] = "High data loss risk"
            suggestions[col] = info
            continue
        
        # 2. Handle numeric columns
        if pd.api.types.is_numeric_dtype(dtype):
            if len(col_data) > 1:  # Ensure we have enough data to calculate stats
                skewness = abs(col_data.skew())
                
                if skewness > skew_threshold:
                    info["suggested_strategy"] = "median"
                    info["rationale"] = f"Data is skewed (skewness: {skewness:.2f})"
                else:
                    info["suggested_strategy"] = "mean"
                    info["rationale"] = f"Data is approximately symmetric (skewness: {skewness:.2f})"
                    
                # Add value suggestions
                info["suggested_value"] = col_data.median() if skewness > skew_threshold else col_data.mean()
            else:
                info["suggested_strategy"] = "constant"
                info["rationale"] = "Not enough data to calculate distribution"
                info["warning"] = "Consider external domain knowledge for fill value"
        
        # 3. Handle datetime columns
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            info["suggested_strategy"] = "interpolate"
            info["rationale"] = "Temporal data often benefits from interpolation"
            info["alternative_strategy"] = "ffill"
        
        # 4. Handle categorical data (objects, categories, booleans)
        else:
            # For categorical data, check if there are rare categories
            value_counts = col_data.value_counts(normalize=True)
            most_common = value_counts.iloc[0]
            most_common_value = value_counts.index[0]
            
            info["suggested_strategy"] = "mode"
            info["suggested_value"] = most_common_value
            info["rationale"] = f"Most frequent category ({most_common:.1%})"
            
            # Check for many rare categories which might indicate a problem
            rare_categories = sum(value_counts < rare_category_threshold)
            if rare_categories > 0:
                info["warning"] = f"{rare_categories} rare categories detected. Consider grouping or 'Other' category"
        
        suggestions[col] = info
    
    # Create a DataFrame and sort by missing percent descending
    result_df = pd.DataFrame.from_dict(suggestions, orient='index')
    if not result_df.empty:
        result_df = result_df.sort_values('missing_percent', ascending=False)
    
    return result_df


def handle_missing_values(df, strategy_dict='auto', constant_value=None, inplace=False):
    """
    Handles missing values in a DataFrame with flexible, per-column strategies.

    This function allows users to specify exactly how to fill nulls for each column.
    If a column is not specified in the strategy_dict, an intelligent default is used.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing missing values.
    strategy_dict : dict, str, or 'auto'
        Specifies the filling strategy.
        - If 'auto': Uses the internal `suggest_fill_strategy` to handle all columns.
        - If a string (e.g., 'mean', 'ffill'): Applies that strategy to all numeric columns.
        - If a dictionary: Maps column names to strategies. 
          Supported strategies: 'mean', 'median', 'mode', 'ffill', 'bfill', 
          'drop', 'constant', 'interpolate'.
    constant_value : dict, optional
        Required if any strategy is 'constant'. Maps column names to the value to use.
        Example: {'price': 0, 'name': 'Unknown'}
    inplace : bool, default False
        If True, modifies the DataFrame in-place. Otherwise, returns a copy.

    Returns:
    --------
    pandas.DataFrame or None
        A copy of the DataFrame with missing values handled, unless inplace=True.

    Raises:
    -------
    ValueError
        If a strategy is not recognized or 'constant' is used without constant_value.

    Example:
    --------
    # Different strategies for different columns
    strategies = {'age': 'median', 'salary': 'mean', 'department': 'mode', 'start_date': 'ffill'}
    df_filled = handle_missing_values(df, strategy_dict=strategies)

    # Use a global strategy for all columns
    df_filled = handle_missing_values(df, strategy_dict='ffill')

    # Fully automatic mode
    df_filled = handle_missing_values(df, strategy_dict='auto')
    """
    
    # Work on a copy if not inplace
    if not inplace:
        df = df.copy()
    
    # If strategy_dict is a string, apply it to all columns with missing values
    if isinstance(strategy_dict, str):
        if strategy_dict == 'auto':
            # Use your previously defined suggestion function
            try:
                suggestions = suggest_fill_strategy(df)
                strategy_dict = suggestions['suggested_strategy'].to_dict()
                # For constant strategy, extract the suggested value
                if constant_value is None:
                    constant_value = {}
                if 'suggested_value' in suggestions.columns:
                    constant_cols = suggestions[suggestions['suggested_strategy'] == 'constant'].index
                    for col in constant_cols:
                        if pd.notna(suggestions.loc[col, 'suggested_value']):
                            constant_value[col] = suggestions.loc[col, 'suggested_value']
            except NameError:
                raise NameError("'suggest_fill_strategy' function is not defined. "
                               "Please define it or use explicit strategies.")
        else:
            # Create a dict with the same strategy for all columns that have missing values
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            strategy_dict = {col: strategy_dict for col in cols_with_missing}
    
    # Ensure constant_value is a dictionary
    if constant_value is None:
        constant_value = {}
    
    # Process each column according to the strategy dictionary
    for col, strategy in strategy_dict.items():
        if col not in df.columns:
            continue  # Skip if column doesn't exist
            
        if not df[col].isnull().any():
            continue  # Skip if no missing values
            
        if strategy == 'drop_column':
            # Drop rows where this specific column is null
            df.drop(columns=[col], inplace=True)

        elif strategy == 'drop_rows':
            df.dropna(subset=[col], inplace=True)
            
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df[col]):
                df.fillna({col : df[col].mean()}, inplace=True)
            else:
                print(f"Warning: Cannot use 'mean' strategy on non-numeric column '{col}'. Skipping.")
                
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df[col]):
                df.fillna({col : df[col].median()}, inplace=True)
            else:
                print(f"Warning: Cannot use 'median' strategy on non-numeric column '{col}'. Skipping.")
                
        elif strategy == 'mode':
            # Handle the case where mode might return multiple values
            mode_values = df[col].mode()
            if not mode_values.empty:
                df.fillna({col : mode_values[0]}, inplace=True)
            else:
                print(f"Warning: Could not calculate mode for column '{col}'. Skipping.")
                
        elif strategy == 'ffill':
            df[col].fillna(method='ffill', inplace=True)
            
        elif strategy == 'bfill':
            df[col].fillna(method='bfill', inplace=True)
            
        elif strategy == 'interpolate':
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].interpolate()
            else:
                print(f"Warning: Cannot use 'interpolate' on non-numeric/non-datetime column '{col}'. Skipping.")
                
        elif strategy == 'constant':
            if col in constant_value:
                df.fillna({col: constant_value[col]}, inplace=True)
            else:
                raise ValueError(f"Constant strategy specified for '{col}' but no constant_value provided.")
                
        else:
            raise ValueError(f"Unknown strategy '{strategy}' for column '{col}'.")
    
    if not inplace:
        return df