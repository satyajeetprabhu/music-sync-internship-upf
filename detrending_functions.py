import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as sp
import pandas as pd


# UITILITY FUNCTIONS

def add_window(df, cycle_column, tolerance, num_div):
    
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
    """
    values = df.values.flatten()  # Flatten the DataFrame to a 1D array
    values = values[~np.isnan(values)]  # Remove NaN values
    closest_value = values[(abs(values - time)).argmin()]  # Find the closest value to `time`
    return closest_value


def find_onsets_in_window(ini_time, end_time, df):
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
def detrend_anchor(meter_df, time_column, instr_df, tolerance=(0.5,0.5), num_div=16, mode=1):
    """
    Adds level 2 m.cycle times to `meter_df` based on Intermediate "instrument anchor" method.
    
    Parameters:
    - meter_df: DataFrame containing the 'Keyboard.Time' column.
    - instr_df: DataFrame containing the onset times of instruments. The order of columns assumes order of preference.
    - tolerance: Parameter for defining the shape of the window (used in mode=2).
    - num_div: Parameter for dividing the window (used in mode=2).
    - mode: Determines which method to use.
      - mode=1: Uses `Closest Instrument.
      - mode=2: Uses `defined.instr.rules`.
    
    Returns:
    - meter_df with the new 'Anchor.Time' column added.
    """
    
    meter_df = meter_df.copy()
    
    if mode == 1:
        # Add Anchor.Time based on the closest onset among all instruments
        meter_df['Anchor.Time.1'] = meter_df[time_column].apply(lambda time: find_closest_onset_in_df(time, instr_df))
        
        # Operations to handle duplicate values
        time_column = np.array(meter_df[time_column])
        anchor_time = np.array(meter_df['Anchor.Time.1'])

        # Identify indices of duplicate values in 'Anchor.Time.1'
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
        meter_df['Anchor.Time.1'] = anchor_time
        
    elif mode == 2:
        
        meter_df = add_window(meter_df, time_column, tolerance, num_div)

        for index, row in meter_df.iterrows():
            time = row[time_column]
            ini_time = row['ini_time']
            end_time = row['end_time']

            # Find the closest onset within the window
            valid_onsets = find_onsets_in_window(ini_time, end_time, instr_df)

            closest_onset = time  # Default to original time if no valid onsets are found
            
            # Iterate over all valid onsets collected from different columns
            for onset_array in valid_onsets:
                if onset_array.size > 0:
                    # Compute the absolute differences between 'time' and each valid onset
                    differences = np.abs(onset_array - time)
                    # Find the onset with the minimum difference
                    closest_onset = onset_array[np.argmin(differences)]
                    break  # Exit after finding the closest onset in the first non-empty array

            meter_df.at[index, 'Anchor.Time.2'] = closest_onset
        
        # Drop the window columns after the operation
        meter_df.drop(columns=['ini_time', 'end_time'], inplace=True)
        
    else:
        raise ValueError("Invalid mode! Use mode=1 or mode=2")

    return meter_df

# ONSET ASSIGNMENT

def assign_onsets_to_cycles(onsets_df, meter_column, tolerance, num_div):
    
    # Create interpolated points
    cycle_starts = meter_column.to_numpy()

    interp_time = [np.linspace(cycle_starts[i], cycle_starts[i + 1], num_div+1)[:-1] for i in range(len(cycle_starts) - 1)]
    interp_time = np.concatenate(interp_time)
    interp_time = np.append(interp_time, cycle_starts[-1])   

    # Create cycle numbers
    cycle_numbers = np.repeat(np.arange(1, len(cycle_starts)), num_div)
    cycle_numbers = np.append(cycle_numbers, len(cycle_starts))

    # Create a dataframe
    new_df = pd.DataFrame({
                            'Cycle': cycle_numbers,
                            'SD': np.append(np.tile(np.arange(1, num_div+1), len(cycle_starts) - 1), 1),
                            'Iso.Time': interp_time       
                         })
    
    # Insert original cycle times at subdivision 1 positions
    new_df['Cycle.Time'] = np.nan  # Initialize the column with NaN
    new_df.loc[new_df['SD'] == 1, 'CycleTime'] = cycle_starts

    # Re-arrange columns
    new_df = new_df[['Cycle','CycleTime','SD','Iso.Time']]

    # Add window columns
    new_df = add_window(new_df, cycle_column='Iso.Time', tolerance=tolerance, num_div=1)

    # Iterate over columns of instrument_df
    for column in onsets_df.columns:
        new_df[column] = np.nan

        # Find the closest onset within the window
        for index, row in new_df.iterrows():
            
            time = row['Iso.Time']
            ini_time = row['ini_time']
            end_time = row['end_time']

            # Find the closest onset within the window
            valid_onsets = find_onsets_in_window(ini_time, end_time, onsets_df[[column]])
            onset_array = valid_onsets[0]

            closest_onset = np.nan  # Default to nan if no valid onsets are found
            
            if onset_array.size > 0:
                # Compute the absolute differences between 'time' and each valid onset
                differences = np.abs(onset_array - time)
                # Find the onset with the minimum difference
                closest_onset = onset_array[np.argmin(differences)]

            new_df.at[index, column] = closest_onset
            
    # Drop the window columns after the operation
    new_df.drop(columns=['ini_time', 'end_time'], inplace=True)
    
    return new_df


# NORMALISE ONSETS

def normalize_onsets_df(df, inst):
    # Create a new DataFrame to store normalized onsets
    df_normalized = df[['Cycle', 'CycleTime', 'SD', 'Iso.Time']].copy()

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
        for col in inst:
            # Filter the rows belonging to the current cycle for the specific column
            cycle_values = df.loc[df['Cycle'] == cycle, col]

            # Divide by the cycle duration to normalize
            normalized_values = (cycle_values - cycle_start_time) / cycle_duration

            # Assign the normalized values back to the appropriate location in the new DataFrame
            df_normalized.loc[df['Cycle'] == cycle, col] = normalized_values
    
    # Set the instrument columns to NaN for the last cycle
    df_normalized.loc[df['Cycle'] == unique_cycles[-1], inst] = np.nan
    # Ensure that there are no infinite values in the DataFrame
    df_normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df_normalized

# PLOT ONSETS

def plot_cycle_onsets(norm_df=None, instr=None, mean_std=True, hist_ons=False, **kwargs):
    '''
    Plot normalized onsets from the dataframe in subplots, one for each instrument.

    Parameters
    ----------
    norm_df : pd.DataFrame
        DataFrame containing normalized onsets with columns 'Cycle', 'SD', 'Iso.Time', 'D1', 'J1', 'J2'.
    mean_std : bool
        If `True`, then mean and std are plotted for each subdivision.
    hist_ons : bool
        If `True`, then a histogram of all onsets is plotted.
    n_bins : int
        Number of bins to use for the histogram.
    fs : int
        Font size.
    kwargs
        Additional keyword arguments to `matplotlib`.

    Returns
    -------
    None
    '''

    # Set default values for kwargs
    kwargs.setdefault('color', 'seagreen')
    kwargs.setdefault('alpha', 0.6)
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 2)
    colors = cm.get_cmap('Set2')

    cycle_nos = sorted(norm_df['Cycle'].unique())
    num_cycles = len(cycle_nos)
    div_nos = sorted(norm_df['SD'].unique())
    num_div = len(div_nos)
    num_plots = len(instr)
           
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

    if num_plots == 1:
        axes = [axes]

    for idx, instrument in enumerate(instr):
        ax = axes[idx]
        color = colors(idx*2) # Get the color for the current instrument
        kwargs['color'] = color

        for cycle in cycle_nos:
            cycle_df = norm_df[norm_df['Cycle'] == cycle]
            onsets = cycle_df[instrument].values
            onsets = onsets[~np.isnan(onsets)]
            
            ax.plot(onsets, (cycle * np.ones(len(onsets))) + shift_value,
                    linestyle='None', **kwargs)
    
        ax.grid(False)

        # x-ticks at every subdivision
        x_ticks = np.linspace(0, 1, num_div+1)
        x_labels = [num+1 for num in range(num_div)]
        ax.set_xticks(x_ticks[:-1])
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.set_yticks([])
        ax.tick_params(length=10, width=1)
        ax.set_xlim(-0.10, 1)
        ax.set_ylim(0, (num_cycles*1.1) + shift_value)

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylabel(r'Cycles $\longrightarrow$', fontsize=14)

        # Add instrument name to the top right of the plot
        ax.text(1, num_cycles, instrument, fontsize=14, color='black', ha='right', va='top')
        
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
                    mu, std = sp.norm.fit(sd_onsets)
                    ax.errorbar(mu, shift_value * 0.9, xerr=std, fmt='.', capsize=1, color='royalblue')
                    ax.axvline(x=mu, ymin=(shift_value * 0.9)/((num_cycles*1.1) + shift_value), ymax= 1, linestyle='--', color='royalblue', linewidth=0.7)
                    ax.text(mu, shift_value/2, "{:3.0f}".format(mu * 100 * num_div/4) + "%",
                            color='royalblue', horizontalalignment='center',
                            verticalalignment='bottom', fontsize=10)
        
    axes[-1].set_xlabel("Metric position within the rhythm cycle", fontsize=14)

    plt.tight_layout()
    plt.show()