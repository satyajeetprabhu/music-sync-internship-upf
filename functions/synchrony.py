import numpy as np
import pandas as pd

def sync_joint_onsets(df=None, instr1=None, instr2=None):
    """
    Calculate the number of joint onsets between two instruments.
    
    Parameters:
        df (pd.DataFrame): Data frame to be processed.
        instr1 (str): Instrument 1 name to be processed.
        instr2 (str): Instrument 2 name to be processed.
    
    Returns:
        int: Number of joint onsets.
    """
    
    ins1 = df[instr1].to_numpy()
    ins2 = df[instr2].to_numpy()
    # Create a boolean index for non-NaN values in both instruments
    ind = ~np.isnan(ins1) & ~np.isnan(ins2)
    len_joint = np.sum(ind) # Count the number of True values
    return len_joint


def sync_sample_paired(df=None, instr1=None, instr2=None, n=0, bootn=None, beat=None, verbose=False):
    """
    Samples and analyzes asynchronies between two instruments' onsets.
    
    Parameters:
        df (pd.DataFrame or dict): Data frame or dictionary of data frames to be processed.
        instr1 (str): Column name for the first instrument.
        instr2 (str): Column name for the second instrument.
        n (int): Number of samples to be drawn from the pool of joint onsets. If 0, do not sample.
        bootn (int): Number of bootstrap samples to draw. Default is 1.
        beat (str): Column name for the beat structure to be included.
        verbose (bool): If True, display the number of shared onsets.
    
    Returns:
        dict: A dictionary containing asynchronies and beat structures.
    """
    
    if isinstance(df, pd.DataFrame):
        # Convert the specified columns to NumPy arrays
        inst1 = df[instr1].to_numpy()
        inst2 = df[instr2].to_numpy()
        beat = df[beat].to_numpy()

        # Calculate the number of joint onsets between the two instruments
        joint_n = sync_joint_onsets(df, instr1=instr1, instr2=instr2)

        # Set default value for bootn if not provided
        if bootn is None:
            bootn = 1

        # Initialize the lists to store asynchronies and beat structures
        D = []
        beat_L = []

        # Single bootstrap case (bootn = 1)
        if bootn == 1:
            # Create a boolean index for non-NaN values in both instruments
            ind = ~np.isnan(inst1) & ~np.isnan(inst2)
            if verbose:
                print(f'onsets in common: {joint_n}')

            # If the number of joint onsets is greater than or equal to n
            if joint_n >= n:
                if n == 0:
                    # Take all joint onsets if n is 0
                    if verbose:
                        print(f'take all onsets: {joint_n}')
                    sample_ind = np.where(ind)[0]
                else:
                    # Randomly sample n indices from the joint onsets
                    sample_ind = np.random.choice(np.where(ind)[0], n, replace=False)
                # Calculate asynchronies and store them in D
                d = inst1[sample_ind] - inst2[sample_ind]
                D = d.tolist()
                beat_L = beat[sample_ind].tolist()
            else:
                # If joint onsets are fewer than n, take the maximum available
                sample_ind = np.where(ind)[0]
                d = inst1[sample_ind] - inst2[sample_ind]
                D = d.tolist()
                beat_L = beat[sample_ind].tolist()

        # Multiple bootstrap case (bootn > 1)
        if bootn > 1:
            # Create a boolean index for non-NaN values in both instruments
            ind = ~np.isnan(inst1) & ~np.isnan(inst2)
            len_joint = np.sum(ind)
            if verbose:
                print(f'onsets in common: {len_joint}')
            if len_joint >= n:
                # Perform bootstrapping bootn times
                for _ in range(bootn):
                    ind = ~np.isnan(inst1) & ~np.isnan(inst2)
                    sample_ind = np.random.choice(np.where(ind)[0], n, replace=False)
                    d = inst1[sample_ind] - inst2[sample_ind]
                    D.extend(d.tolist())
                    beat_L.extend(beat[sample_ind].tolist())
            else:
                # If joint onsets are fewer than n, take the maximum available
                for _ in range(bootn):
                    ind = ~np.isnan(inst1) & ~np.isnan(inst2)
                    sample_ind = np.random.choice(np.where(ind)[0], n, replace=False)
                    d = inst1[sample_ind] - inst2[sample_ind]
                    D.extend(d.tolist())
                    beat_L.extend(beat[sample_ind].tolist())

    else:
        # Raise Error if df is not a DataFrame
        raise ValueError('df must be a DataFrame')
    
        # This section is commented out because it is not used in the current implementation
        '''
        # If df is a dictionary of DataFrames, process each DataFrame
        NAMES = df.keys()
        D = []
        L = []
        for key in NAMES:
            tmp_df = df[key]
            # Call sync_sample_paired recursively for each DataFrame
            result = sync_sample_paired(df=tmp_df, instr1=instr1, instr2=instr2, n=n, bootn=bootn, beat=beat)
            beat_L = None
            L.append(len(result['asynch']))
            D.append(result)
        
        # Combine results from all DataFrames into a single DataFrame
        R3 = pd.concat(D, ignore_index=True)
        NAME_INDEX = [name for name, length in zip(NAMES, L) for _ in range(length)]
        R3['name'] = NAME_INDEX
        D = R3
        '''

    # Create a DataFrame from the asynchronies and beat structures
    result_df = pd.DataFrame({'asynch': D, 'beatL': beat_L})
    return result_df

