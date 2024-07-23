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



def extract_cycle(annotation):
    
    annotation = [str(x) for x in annotation]
    
    if all(x.isdigit() for x in annotation):
        # return all indices if annotation is a list of integers
        return list(range(len(annotation)))

    else:
        downbeat_pattern = r"^1$|:1$|\|1$|:01$|\.1$|:01:\d{2}$" # Match patterns like 1, :1, |1, :01, .1, :01:xx etc.
        
        # Find indices of elements matching the downbeat pattern
        downbeat_ind = [i for i, x in enumerate(annotation) if re.search(downbeat_pattern, x)]

        return downbeat_ind
    

def add_annotation(df=None, annotation=None, time=None, reference=None):
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

    # Convert annotation and time to lists if they are pandas Series
    if isinstance(annotation, pd.Series):
        annotation = annotation.tolist()
    if isinstance(time, pd.Series):
        time = time.tolist()

    # Extract only the cycle start times from the annotation
    anno_ind = extract_cycle(annotation)    # Find indices of cycle beginnings in the annotation
    cycle_time = [time[i] for i in anno_ind] # Extract cycle start times from the annotated times
    
    # Extract the reference column from the dataframe
    tmp = df[reference]

    # Find indices of elements matching the downbeat pattern
    cycle_ind = extract_cycle(tmp)

    # Check if the number of cycle beginnings are same in annotation and the reference column
    if len(anno_ind) != len(cycle_ind):
        raise ValueError('Error: Cycle beginnings and the annotation cycle data are of different lengths!')


    df = df.copy()

    # If df does not contain a 'Cycle' column already, add it
    
    if not 'Cycle' in df.columns:

        # Define the pattern to remove the suffixes
        pattern = r"(:\d+|\|\d+|\.\d+|:\d+:\d+)$"                   # Match patterns like :xx, |xx, .xx, :xx:xx etc.
        #pattern = r"(:[0-9]+|\|[0-9]+|\.[0-9]+|:[0-9]+:[0-9]+)$"   # Similar to the above pattern but with explicit digits

        # Add the cycle number to the dataframe
        df.loc[:, 'Cycle'] = tmp.str.replace(pattern, '', regex=True).astype(int) # Remove the suffixes and convert to integer
   
    # Add the cycle time to the dataframe
    df.loc[:, 'CycleTime'] = np.nan
    df.loc[cycle_ind, 'CycleTime'] = cycle_time

    return df