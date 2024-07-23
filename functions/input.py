import pandas as pd

def get_df_csv(file_path, delimiter=',', sel_col=None, concat_axis=0, ignore_index=False):
    """
    Load dataframe from text (csv or txt) files.

    Parameters
    ----------
    file_path : str or list of strings
        Names (including paths) of the input files.
    delimiter : str
        String used as delimiter in the input files. Default is ','.
    sel_col : list of int or list of str, optional
        Select the columns of the input files by numbers or names. Default is all columns.
    concat_axis : int, optional
        Axis to concatenate along. 0 to concatenate vertically (default), 1 to concatenate horizontally.
    ignore_index : bool, optional
        If False, continue the index values on the concatenation axis. Default is False.

    Returns
    -------
    df : DataFrame
        Concatenated DataFrame with the selected columns from the input file(s).
    """
    
    # Ensure labels_files is a list
    if isinstance(file_path, str):
        file_path = [file_path]
    
    # Initialize an empty list to store dataframes
    dataframes = []

    for file in file_path:
        # Check for valid file types
        if not (file.endswith('.csv') or file.endswith('.txt')):
            raise ValueError(f"Unsupported file type: {file}. Only .csv and .txt files are supported.")
        
        try:
            # Load the file into a DataFrame
            df = pd.read_csv(file, delimiter=delimiter)
            
            # Select specified columns if sel_col is provided
            if sel_col is not None:
                df = df[sel_col]
            
            # Append the DataFrame to the list
            dataframes.append(df)
        except Exception as e:
            raise RuntimeError(f"Error reading {file}: {e}. Check if the file exists and is correctly formatted.")
    
    # Concatenate all DataFrames in the list
    try:
        concatenated_df = pd.concat(dataframes, axis=concat_axis, ignore_index=ignore_index)
    except Exception as e:
        raise RuntimeError(f"Error concatenating dataframes: {e}")

    return concatenated_df

'''
To do: 
Incorporate code to input sel_col as [:5], [2:10] etc. to select columns by slicing.
'''