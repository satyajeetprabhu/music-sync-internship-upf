"""
Detrending
===========

Detrending cycle starts to align with the onsets of one or more instruments
-------------------------------

Input
-----
.. autosummary::
    :toctree: generated/

    load_df_csv

Utility functions
-----------------
.. autosummary::
    :toctree: generated/
    
    add_window
    find_closest_onset_in_df
    find_onsets_in_window

Detrending
----------
.. autosummary::
    :toctree: generated/
        
    detrend_anchor
    assign_onsets_to_cycles
    normalize_onsets_df

Plotting
--------
.. autosummary::
    :toctree: generated/

    plot_cycle_onsets
    
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as sp
import pandas as pd


__all__ = ['load_df_csv', 'add_window', 'find_closest_onset_in_df', 'find_onsets_in_window', 'detrend_anchor', 'assign_onsets_to_cycles', 'normalize_onsets_df', 'plot_cycle_onsets']


# INPUT

def load_df_csv(file_path, delimiter=',', sel_col=None, concat_axis=0, ignore_index=False, header='infer'):
    """
    Load dataframe from text (csv or txt) files.

    Parameters
    ----------
    file_path : str or list of strings
        names (including paths) of the input files.
    delimiter : str
        string used as delimiter in the input files. Default is ','.
    sel_col : list of int or list of str
        select the columns of the input files by numbers or names. Default is all columns.
    concat_axis : int
        axis to concatenate along. 0 to concatenate vertically (default), 1 to concatenate horizontally.
    ignore_index : bool
        if False, continue the index values on the concatenation axis. If True, reset the index. Default is False. 
    header : int, 'infer' or 'None'
        row number to use as the column names. Default is 'infer'. 'None' means no header.

    Returns
    -------
    df : DataFrame
        Concatenated DataFrame with the selected columns from the input file(s).

    Notes
    -----
    To do: 
    Incorporate code to input sel_col as [:5], [2:10] etc. to select columns by slicing.
    """
    
    # Ensure file_path is a list
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
            df = pd.read_csv(file, delimiter=delimiter, header=header)
            
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


# UITILITY FUNCTIONS

def add_window(df, cycle_column, tolerance=(0.5,0.5), num_div=16):
    
    """
    Adds 'ini_time' and 'end_time' columns to the DataFrame `df` by constructing window around
    the cycle column based on tolerance values and number of subdivisions in a cycle. Used 
    to select candidate onsets for detrending and assignment of onsets to subdivisions.

    Parameters
    ----------
    - df : DataFrame 
        dataFrame containing the cycle start times.
    - cycle_column: str
        name of the column in df containing cycle start times.
    - tolerance: tuple
        tuple containing the left and right tolerance values that define the shape of
        the window around cycle start time. Default is (0.5, 0.5).
    - num_div: Number of subdivisions in the cycle.

    Returns: 
    -------
    - df : DataFrame
        original DataFrame with 'ini_time' and 'end_time' columns added.

    Notes
    -----
    - The tuple `tolerance` is interpreted as follows:
        - tolerance[0] is the left tolerance value
        - tolerance[1] is the right tolerance value
    - The recommended values for tolerance are between 0 and 1. For consistency, the sum of 
        the two values should be 1 but this is left to the discretion of the user.
    - The window is constructed as follows:
        ini_time = cycle_column - (subivision duration L * tolerance[0])
        end_time = cycle_column + (subdivision duration R * tolerance[1])
    - For example (0.5, 0.5) implies that the window should cover half the duration of a subdivision 
        on either side of the cycle start time.
    """

    df = df.copy()
    
    # Window of tolerance for left and right of the cycle annotation
    toleranceL = (1/num_div) * tolerance[0]
    toleranceR = (1/num_div) * tolerance[1]

    # Calculate cycle lengths on either side of keyboard times
    df['cycle_dur_L'] = df[cycle_column].diff()  # iloc(n) - iloc(n-1)
    df['cycle_dur_R'] = df['cycle_dur_L'].shift(-1)  # iloc(n+1) - iloc(n)

    # Handle the NaN values by filling with the nearest available value
    df['cycle_dur_L'].fillna(df['cycle_dur_R'], inplace=True)  # First entry: copy value from cycle_dur_R
    df['cycle_dur_R'].fillna(df['cycle_dur_L'], inplace=True)  # Last entry: copy value from cycle_dur_L

    # Create window on the left and right side of the keyboard time
    df['ini_time'] = df[cycle_column] - (df['cycle_dur_L'] * toleranceL)
    df['end_time'] = df[cycle_column] + (df['cycle_dur_R'] * toleranceR)

    df.drop(columns=['cycle_dur_L', 'cycle_dur_R'], inplace=True)

    return df


def find_closest_onset_in_df(time, df):
    """
    Finds the closest value to `time` in the entire DataFrame `df`.
    Ignores NaN values.

    Parameters
    ----------
    - time : float
        The time value to search for.
    - df : DataFrame
        The DataFrame to search in.

    Returns
    -------
    - closest_value : float
        The closest value to `time` in `df`.
    """
    values = df.values.flatten()  # Flatten the DataFrame to a 1D array
    values = values[~np.isnan(values)]  # Remove NaN values
    closest_value = values[(abs(values - time)).argmin()]  # Find the closest value to `time`
    return closest_value


def find_onsets_in_window(df, ini_time, end_time):
    """
    Finds valid onsets in the DataFrame `df` that fall within the window defined by `ini_time` and `end_time`.

    Parameters
    ----------
    - df : DataFrame
        The DataFrame containing the onset times.
    - ini_time : float
        The start time of the window.
    - end_time : float
        The end time of the window.

    Returns
    -------
    - valid_onsets_list : list
        A list of arrays containing the valid onsets for each column in `df`. 
        If no valid onsets are found in a column, an empty array is returned for that column.

    """
    valid_onsets_list = []
    
    for column in df.columns:
        valid_onsets = df[column].dropna().loc[(df[column] >= ini_time) & (df[column] < end_time)].values
        
        if valid_onsets.size > 0:
            valid_onsets_list.append(valid_onsets)
        else:
            valid_onsets_list.append(np.array([]))
    
    return valid_onsets_list


# DETRENDING

# Main function to add Anchor.Time columns to meter_df based on mode

def detrend_anchor(meter_df, time_column, instr_df, tolerance=(0.5,0.5), num_div=16, mode='closest.instr', col_name='Anchor.Time'):
    """
    Adds level 2 m.cycle times to `meter_df` based on Intermediate "instrument anchor" method.
    
    Parameters:
    -----------
    - meter_df: DataFrame
        DataFrame containing the cycle start times.
    - time_column: str
        name of the column in meter_df containing cycle start times.
    - instr_df: DataFrame
        DataFrame containing the onset times of instruments. The order of columns assumes order of preference.
    - tolerance: tuple
        parameter for defining the shape of the window (used when mode='defined.instr').
    - num_div: int
        Number of subdivisions in a cycle (used when mode='defined.instr').
    - mode: str
        Determines which method to use.
      - mode = 'closest.instr': Uses `Closest Instrument.
      - mode = 'defined.instr': Uses `defined.instr.rules`.
    - col_name: str
        Name of the new column to be added to meter_df. Default is 'Anchor.Time'.
    
    Returns:
    - meter_df: DataFrame
        original meter_df with the new 'Anchor.Time' column added.

    Notes:
    ------
    - In mode='defined.instr', we could use find_closest from Carat utils post integration

    """
    
    meter_df = meter_df.copy()
    
    if mode == 'closest.instr':
        
        # Add Anchor.Time based on the closest onset among all instruments
        meter_df[col_name] = meter_df[time_column].apply(lambda time: find_closest_onset_in_df(time, instr_df))
        
        # Operations to handle duplicate values
        time_column = np.array(meter_df[time_column])
        anchor_time = np.array(meter_df[col_name])

        # Identify indices of duplicate values in 'Anchor.Time'
        unique, counts = np.unique(anchor_time, return_counts=True)
        duplicate_values = unique[counts > 1]

        # Initialize a mask that will determine which rows to modify
        mask = np.zeros_like(anchor_time, dtype=bool)

        # For each duplicate value, find the row with the smallest time difference
        for value in duplicate_values:
            indices = np.where(anchor_time == value)[0]
            time_diffs = np.abs(time_column[indices] - anchor_time[indices])
            min_index = indices[np.argmin(time_diffs)]
            
            # Update the mask to mark all but the closest one for modification
            mask[indices] = True
            mask[min_index] = False

        # Modify the `anchor_time` array based on the mask
        anchor_time[mask] = time_column[mask]

        # Update the DataFrame
        meter_df[col_name] = anchor_time
        
    elif mode == 'defined.instr':
        
        # Add window columns to meter_df
        meter_df = add_window(meter_df, time_column, tolerance, num_div)

        # Iterate over rows of meter_df
        for index, row in meter_df.iterrows():
            time = row[time_column]
            ini_time = row['ini_time']
            end_time = row['end_time']

            # Find the closest onset within the window
            valid_onsets = find_onsets_in_window(df=instr_df, ini_time=ini_time, end_time=end_time)

            # Default to original time if no valid onsets are found
            closest_onset = time  
            
            # Iterate over all valid onsets collected from different columns
            for onset_array in valid_onsets:
                if onset_array.size > 0:
                    # Compute the absolute differences between 'time' and each valid onset
                    differences = np.abs(onset_array - time)
                    # Find the onset with the minimum difference
                    closest_onset = onset_array[np.argmin(differences)]
                    break  # Exit after finding the closest onset in the first non-empty array

            meter_df.at[index, col_name] = closest_onset
        
        # Drop the window columns after the operation
        meter_df.drop(columns=['ini_time', 'end_time'], inplace=True)
        
    else:
        raise ValueError("Invalid mode! Use mode='closest.instr' or mode='defined.instr'")

    return meter_df


def assign_onsets_to_cycles(onsets_df, cycle_time, tolerance=(0.5,0.5), num_div=16):
    
    """
    Assigns onsets to subdivisions in the cycle based on the closest onset within a window
    and constructs a new dataframe that follows the structure of IEMP selected_onsets df.
    Useful for further synchrony analysis and visualization of onsets.

    Parameters
    ----------
    - onsets_df : DataFrame
        DataFrame with instrument columns containing their respective onset times.
    - cycle_time : DataFrame column (Pandas Series) or numpy array
        Column or array containing cycle start times.
    - tolerance : tuple
        Parameter for defining the shape of the window. Default is (0.5, 0.5).
    - num_div : int
        Number of subdivisions in a cycle. Default is 16.

    Returns
    -------
    - new_df : DataFrame
        DataFrame containing the columns in the format followed by IEMP: 
        cycle numbers, cycle start times, subdivision numbers, isochronous subdivision 
        times and instrument columns with onsets assigned to subdivisions.

    Notes
    -----
    - find_closest from Carat utils can also be used after integration
    """
    
    #  Check if cycle_time is a DataFrame column or a numpy array
    if isinstance(cycle_time, pd.Series):
        cycle_starts = cycle_time.to_numpy()

    # Create isochronous grid of time points between cycle starts
    iso_time = [np.linspace(cycle_starts[i], cycle_starts[i + 1], num_div+1)[:-1] for i in range(len(cycle_starts) - 1)]
    iso_time = np.concatenate(iso_time)
    iso_time = np.append(iso_time, cycle_starts[-1])   # Add the last cycle start time

    # Create cycle numbers
    cycle_numbers = np.repeat(np.arange(1, len(cycle_starts)), num_div)
    cycle_numbers = np.append(cycle_numbers, len(cycle_starts)) # Add the last cycle number

    # Create subdivision numbers
    sub_div = np.tile(np.arange(1, num_div+1), len(cycle_starts) - 1)
    sub_div = np.append(sub_div, 1) # Assign subdivision 1 to the last cycle start time

    # Create a dataframe
    new_df = pd.DataFrame({
                            'Cycle': cycle_numbers,
                            'SD': sub_div,
                            'Iso.Time': iso_time       
                         })
    
    # Insert original cycle times at subdivision 1 positions
    new_df['Cycle.Time'] = np.nan  # Initialize the column with NaN
    new_df.loc[new_df['SD'] == 1, 'CycleTime'] = cycle_starts

    # Re-arrange columns
    new_df = new_df[['Cycle','CycleTime','SD','Iso.Time']]

    # ASSIGNING ONSETS TO CYCLES

    # Add windows around every subdivision
    new_df = add_window(new_df, cycle_column='Iso.Time', tolerance=tolerance, num_div=1)

    # Add columns for each instrument in onsets_df
    for column in onsets_df.columns:
        new_df[column] = np.nan

    # Iterate through each row in new_df
    for index, row in new_df.iterrows():

        time = row['Iso.Time']
        ini_time = row['ini_time']
        end_time = row['end_time']

        # Find the valid onsets within the window for each instrument
        valid_onsets_list = find_onsets_in_window(df=onsets_df, ini_time=ini_time, end_time=end_time)

        # Iterate through each instrument in onsets_df
        for inst, valid_onset_array in zip(onsets_df.columns, valid_onsets_list):

            closest_onset = np.nan  # Default to nan if no valid onsets are found

            if valid_onset_array.size > 0:
                # Compute the absolute differences between 'time' and each valid onset
                differences = np.abs(valid_onset_array - time)
                # Find the onset with the minimum difference
                closest_onset = valid_onset_array[np.argmin(differences)]

            # Assign the closest onset to the corresponding column in new_df
            new_df.at[index, inst] = closest_onset

    # Drop the window columns after the operation
    new_df.drop(columns=['ini_time', 'end_time'], inplace=True)
    
    return new_df


def normalize_onsets_df(df, instr):
    """
    Normalize each instrument onset time in the DataFrame `df` to the cycle duration.
    The normalization is done by dividing the onset time by the cycle duration.

    Parameters
    ----------
    - df : DataFrame
        DataFrame containing the instrument onset times.
    - inst : str or list of str
        List of instrument columns to normalize.

    Returns
    -------
    - df_normalized : DataFrame
        Original DataFrame with added 'instr_norm' columns containing normalized onset times for 
        each instrument.

    Notes
    -----
    - This function currently expects the DataFrame to have columns 'Cycle', 'Iso.Time' and instrument 
        columns as generated by the assign_onsets_to_cycles() function.
        
    """
    # Create a new DataFrame to store normalized onsets
    df_normalized = df.copy()

    if isinstance(instr, str):
        instr = [instr]

    # Get the unique cycles
    unique_cycles = df['Cycle'].unique()
    
    # Iterate over each cycle except the last one
    for cycle in unique_cycles[:-1]:
        # Filter the rows belonging to the current cycle
        cycle_df = df[df['Cycle'] == cycle]
        
        # Get the cycle start and end times
        cycle_start_time = cycle_df['Iso.Time'].iloc[0]
        next_cycle_start_time = df[df['Cycle'] == cycle + 1]['Iso.Time'].iloc[0]
        
        # Calculate the cycle duration
        cycle_duration = next_cycle_start_time - cycle_start_time
        
        # Normalize each instrument column in instr
        for col in instr:
            # Filter the rows belonging to the current cycle for the specific column
            cycle_values = df.loc[df['Cycle'] == cycle, col]

            # Divide by the cycle duration to normalize
            normalized_values = (cycle_values - cycle_start_time) / cycle_duration

            # Create a new column name with '_norm' suffix
            norm_col_name = f"{col}_norm"
            
            # Assign the normalized values to the new column in the DataFrame
            df_normalized.loc[df['Cycle'] == cycle, norm_col_name] = normalized_values
    
    # Set the instrument columns with '_norm' suffix to NaN for the last cycle
    for col in instr:
        norm_col_name = f"{col}_norm"
        df_normalized.loc[df['Cycle'] == unique_cycles[-1], norm_col_name] = np.nan

    # Ensure that there are no infinite values in the DataFrame
    df_normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df_normalized


# PLOT ONSETS

def plot_cycle_onsets(df=None, instr=None, mean_std=True, hist_ons=False, **kwargs):
    '''
    Plot normalized onsets from the dataframe in subplots, one for each instrument.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing instrument onsets and columns 'Cycle' and 'SD'.
    instr : str or list of str
        List of instrument names to plot.
    mean_std : bool
        If `True`, then mean and std are plotted for each subdivision.
    hist_ons : bool
        If `True`, then a histogram of all onsets is plotted.
    kwargs
        Additional keyword arguments to `matplotlib`.

    Returns
    -------
    None

    Notes
    -----
    - Need to find a more robust calculation to shift the plots based on different data sizes
    '''

    # Check if instr is a string or a list
    if isinstance(instr, str):
        instr = [instr]

    # Add '_norm' suffix to instrument names
    instr_norm = [f"{inst}_norm" for inst in instr]   

    # Normalize onsets
    norm_df = normalize_onsets_df(df, instr)

    # Set default values for kwargs
    kwargs.setdefault('color', 'seagreen')
    kwargs.setdefault('alpha', 0.6)
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 2)
    colors = cm.get_cmap('Set2')

    # Get number of cycles, subdivisions and plots
    cycle_nos = sorted(norm_df['Cycle'].unique())
    num_cycles = len(cycle_nos)
    div_nos = sorted(norm_df['SD'].unique())
    num_div = len(div_nos)
    num_plots = len(instr)

    # Set shift value for bottom margin based on the plot type       
    if hist_ons:
        shift_value = num_cycles * 0.4
    elif mean_std:
        shift_value = num_cycles * 0.3
    else:
        shift_value = num_cycles * 0.1
    
    # Calculate the total height of the figure
    total_height = 3 * num_plots

    # Create a figure with a subplot for each instrument
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, total_height), sharex=True)

    # Ensure axes is a list
    if num_plots == 1:
        axes = [axes]

    # Plot onsets for each instrument using their normalized values
    for idx, instrument in enumerate(instr_norm):
        # Get the current axis
        ax = axes[idx]
        
        # Set the color for the current instrument
        color = colors(idx*2) # Get a color from the colormap
        kwargs['color'] = color

        # Plot onsets for each cycle
        for cycle in cycle_nos:
            cycle_df = norm_df[norm_df['Cycle'] == cycle]
            onsets = cycle_df[instrument].values
            onsets = onsets[~np.isnan(onsets)]
            
            # Plot the onsets for the current cycle
            ax.plot(onsets, (cycle * np.ones(len(onsets))) + shift_value,
                    linestyle='None', **kwargs)
    
        # Set the plot properties
        ax.grid(False)
        ax.tick_params(length=10, width=1)

        # Set the x-axis ticks and labels
        x_ticks = np.linspace(0, 1, num_div+1)
        x_labels = [num+1 for num in range(num_div)]
        ax.set_xticks(x_ticks[:-1])
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.set_xlim(-0.10, 1)

        # Set the y-axis ticks and labels
        ax.set_yticks([])
        ax.set_ylim(0, (num_cycles*1.1) + shift_value)
        ax.set_ylabel(r'Cycles $\longrightarrow$', fontsize=14)
           
        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add instrument name to the top right of the plot
        ax.text(1, num_cycles, instr[idx], fontsize=14, color='black', ha='right', va='top')
        
        if hist_ons:
            # Flatten onsets for histogram
            onsets_flattened = norm_df[instrument].values.flatten()
            onsets_flattened = onsets_flattened[~np.isnan(onsets_flattened)]
            ax.hist(onsets_flattened, bins=100, density=False, alpha=0.2, facecolor='black')

        if mean_std:
            # Calculate and plot mean/std dev for each SD value
            for sd in div_nos:
                sd_onsets = norm_df[norm_df['SD'] == sd][instrument].values.flatten()
                sd_onsets = sd_onsets[~np.isnan(sd_onsets)]
                
                if len(sd_onsets) > 0:
                    # Fit a normal distribution to the data
                    mu, std = sp.norm.fit(sd_onsets)
                    
                    # Plot the mean and std dev for the current SD value
                    ax.errorbar(mu, shift_value * 0.9, xerr=std, 
                                fmt='.', capsize=1, color='royalblue')
                    ax.axvline(x=mu, 
                               ymin=(shift_value * 0.9)/((num_cycles*1.1) + shift_value), 
                               ymax= 1, linestyle='--', color='royalblue', linewidth=0.7)
                    ax.text(mu, shift_value/2, "{:3.0f}".format(mu * 100 * num_div/4) + "%",
                            color='royalblue', horizontalalignment='center',
                            verticalalignment='bottom', fontsize=10)
        
    axes[-1].set_xlabel("Metric position within the rhythm cycle", fontsize=14)

    plt.tight_layout()
    plt.show()