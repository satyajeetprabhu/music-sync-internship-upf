import pandas as pd
import re
import numpy as np


def add_beatsd(df=None, num_div=4, sd_column='SD'):
    """
    Add Beat.SD annotations to a dataframe based on specified subdivisions.
    
    Args:
    df (pd.DataFrame): The input dataframe containing the subdivisions.
    num_div (int): The number of subdivisions per cycle (default is 4).
    sd_column (str): The name of the column containing subdivision values (default is 'SD').

    Returns:
    pd.DataFrame: A new dataframe with the added 'Beat.SD' annotations.
    """

    # Check if reference column exists in the dataframe
    if sd_column not in df.columns:
        raise ValueError(f"Error: reference '{sd_column}' does not exist in the data frame.")

    # Check if the number of divisions is valid
    if num_div < 1 and not isinstance(num_div, int):
        raise ValueError(f"Error: number of divisions must be an integer greater than 0.")
    
    # Raise an error if the column does not contain only integers
    if not (pd.api.types.is_integer_dtype(df[sd_column])):
        raise ValueError("Error: reference column must contain only integers.")
    
    # Check if num_div equally divides the number of subdivisions
    max_value = df[sd_column].max()
    if max_value % num_div != 0:
        raise ValueError(f"Error: number of divisions '{num_div}' does not equally divide the {max_value} subdivisions in the '{sd_column}' column.")


    df = df.copy()

    # Function to generate Beat.SD annotations
    def generate_annotations(x, num_div):
        cycle = (x - 1) // num_div + 1
        beat = (x - 1) % num_div + 1
        return f"{cycle}.{beat}"

    # Apply the function to generate the new column
    df['Beat.SD'] = df[sd_column].apply(lambda x: generate_annotations(x, num_div))

    return df


def add_cycle(df, sd_column='SD', cycle_column='Cycle'):
    """
    Add a 'Cycle' column to the DataFrame based on the 'SD' column values. Accepts SD, Beat.SD and label SD formats.

    Parameters:
    - df: DataFrame containing the 'SD' column. 
    - sd_column: Name of the column containing the 'SD' values.
    - cycle_column: Name of the column to be added for cycle numbers.

    Returns:
    - DataFrame with the 'Cycle' column added.
    """

    if cycle_column in df.columns:
        raise ValueError('Cycle column already exists in dataframe. Select a different column name.')
    
    # Copy the SD column as a new df tmp
    tmp = df.copy()
    
    # For the specific case of 'Label SD' or 'Label.SD' format used in IEMP data
    if sd_column == 'Label SD' or sd_column == 'Label.SD':
        
        ref = tmp[sd_column]
        # Define the pattern to remove the suffixes
        pattern = r"(:\d+|\|\d+|\.\d+|:\d+:\d+)$"                   # Match patterns like :xx, |xx, .xx, :xx:xx etc.
        #pattern = r"(:[0-9]+|\|[0-9]+|\.[0-9]+|:[0-9]+:[0-9]+)$"   # Similar to the above pattern but with explicit digits

        # Add the cycle number to the dataframe
        tmp['Cycle'] = ref.str.replace(pattern, '', regex=True).astype(int) # Remove the suffixes and convert to integer
        
        return tmp
    
    # For the general case of SD (1, 2, 3, ...) or Beat.SD (1.1, 1.2, 1.3, ...) formats
    else:
        # Get unique beats and sort them
        unique_beats = sorted(tmp[sd_column].unique())

        # Create a mapping from beat values to numbers
        beat_mapping = {beat: i + 1 for i, beat in enumerate(unique_beats)}

        # Map the beat values to their corresponding numbers
        tmp['Mapped'] = tmp[sd_column].map(beat_mapping)

        # Initialize variables
        current_cycle = 1
        cycle_numbers = []
        pattern = list(range(1, len(unique_beats) + 1))
        pattern_index = 0

        # Iterate through the mapped beats and assign cycle numbers
        for beat in tmp['Mapped']:
            if beat == pattern[pattern_index]:
                cycle_numbers.append(current_cycle)
                pattern_index += 1
                if pattern_index == len(pattern):
                    pattern_index = 0
                    current_cycle += 1
            else:
                # Handle the case where beat does not match the expected pattern
                # Reset pattern_index and re-check the current beat
                while beat != pattern[pattern_index]:
                    pattern_index += 1
                    if pattern_index == len(pattern):
                        pattern_index = 0
                        current_cycle += 1
                cycle_numbers.append(current_cycle)
                pattern_index += 1
                if pattern_index == len(pattern):
                    pattern_index = 0
                    current_cycle += 1

        # Assign the cycle numbers to the tmp DataFrame
        tmp[cycle_column] = cycle_numbers

        # Drop the 'Mapped' column as it was only needed for the mapping process
        tmp = tmp.drop(columns=['Mapped'])

        return tmp


def extract_cycle(annotation):
    '''
    Extracts the indices of cycle beginnings from the annotation.

    Args:
    annotation (list): Annotation to be processed

    Returns:
    list: Indices of cycle beginnings
    '''
    annotation = [str(x) for x in annotation]
    
    if all(x.isdigit() for x in annotation):

        if len(annotation) == len(set(annotation)):
           #print('1')
           return list(range(len(annotation))) # Return all indices if all elements are digits 
        
        else:
            downbeat_ind = [0]  # The first index is always a starting point

            for i in range(1, len(annotation)):
                if annotation[i] != annotation[i - 1]:
                    downbeat_ind.append(i)
            #print('2')
            return downbeat_ind

    else:
        downbeat_pattern = r":1$|\|1$|:01$|\.1$|:01:\d{2}$" # Match patterns like :1, |1, :01, .1, :01:xx etc.
        # Find indices of elements matching the downbeat pattern
        downbeat_ind = [i for i, x in enumerate(annotation) if re.search(downbeat_pattern, x)]

        #print('3')
        return downbeat_ind
    

def add_annotation(df=None, reference=None, annotation=None, time=None):
    """
    Adds cycle start times from annotations to onsets.

    Args:
        df (pd.DataFrame): Data frame to be processed (required)
        annotation (list or pandas Series): Annotation (cycle starts) to be added
        time (list or pandas Series): Annotated times of the cycle starts to be added
        reference (str): The target column for cycles in IEMP data

    Returns:
        DataFrame: Dataframe with added column of cycles and cycle onsets
    """
    # Check if reference column exists in the dataframe
    if reference not in df.columns:
        raise ValueError(f"Error: reference '{reference}' does not exist in the data frame.")

    ### Check if the number of cycles in annotation and reference match before proceeding further

    # Extract only the cycle start times from the annotation and reference column
    anno_ind = extract_cycle(annotation)    # Find indices of cycle beginnings in the annotation
    cycle_ind = extract_cycle(df[reference])    # Find indices of cycle beginnings in the reference column

    # Check if the number of cycle beginnings are same in annotation and the reference column
    if len(anno_ind) != len(cycle_ind):
        raise ValueError('Error: Cycle beginnings and the annotation cycle data are of different lengths!')

    df = df.copy()

    if not 'Cycle' in df.columns:
        df = add_cycle(df, sd_column=reference)

    # Add the cycle time to the dataframe
    df.loc[:, 'CycleTime'] = np.nan
    cycle_time = [time[i] for i in anno_ind] # Extract cycle start times from the annotated times
    df.loc[cycle_ind, 'CycleTime'] = cycle_time

    return df


def add_isobeats(df, instr, beat, beatlabel='Iso.Time'):
    """
    Add isochronous times or mean onset times to the data frame.

    Parameters:
    df (pd.DataFrame): Data frame to be processed.
    instr (list): List of instruments for constructing the reference time.
    beat (str): Column name for beat sub-divisions. Takes both 'SD' and 'Beat.SD' format.
    beatlabel (str): Column name for the newly created beats. Default is 'Iso.Time'.

    Returns:
    pd.DataFrame: Data frame with added column of isochronous beats or mean onset times.
    """
    
    df = df.copy()

    if len(instr) > 1:
        df['Mean.Time'] = df[instr].mean(axis=1, skipna=True)
        if beatlabel != 'Iso.Time':
            df.rename(columns={'Mean.Time': beatlabel}, inplace=True)
        return df

    if len(instr) == 1:
        
        # Get unique beats and sort them
        unique_beats = sorted(df[beat].unique())
        # Create a mapping from beat values to numbers
        beat_mapping = {beat: i + 1 for i, beat in enumerate(unique_beats)}
        # Map the beat values to their corresponding numbers
        df['beat_num'] = df[beat].map(beat_mapping)
        
        beat_N = df['beat_num'].max(skipna=True)
        
        df = add_cycle(df, sd_column=beat, cycle_column='cycle')
        df['mean_onset'] = df[instr].mean(axis=1, skipna=True)

        iso_times = []
        for k in range(1, int(df['cycle'].max(skipna=True))):
            tmp = df[df['cycle'] == k]
            tmp_next = df[df['cycle'] == k + 1]
            if tmp['mean_onset'].isna().iloc[0] or tmp_next['mean_onset'].isna().iloc[0]:
                s = [np.nan] * beat_N
            else:
                s = np.linspace(tmp['mean_onset'].iloc[0], tmp_next['mean_onset'].iloc[0], beat_N + 1)[:-1]
            iso_times.extend(s)

        # Handle the last incomplete cycle if present
        last_cycle = df[df['cycle'] == df['cycle'].max(skipna=True)]
        if not last_cycle.empty:
            iso_times.extend([np.nan] * len(last_cycle))

        df[beatlabel] = iso_times[:len(df)]
        df.drop(columns=['cycle', 'mean_onset', 'beat_num'], inplace=True)

        return df
