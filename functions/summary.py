import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp

def summarise_onsets(df=None, instr=None, filter_lower=0, filter_upper=2, binw=0.25, plot=False):
    
    # Empty variables
    instrument = None
    ungroup = None
    time_diff = None
    d = None
    SUMMARY = np.zeros((len(instr), 6))
    HIS = pd.DataFrame()

    # Loop across instruments
    for k in range(len(instr)):
        # Select instrument column from dataframe
        tmp = df[[instr[k]]].copy()
        tmp.columns = ['instrument']

        # Calculate time differences and filter
        h = tmp.dropna().copy()
        h['time_diff'] = h['instrument'] - h['instrument'].shift(1)
        h = h[(h['time_diff'] >= filter_lower) & (h['time_diff'] <= filter_upper)]

        # Append results to HIS dataframe
        HIS = pd.concat([HIS, pd.DataFrame({'d': h['time_diff']*1000, 'Instrument': instr[k]})])

        # Calculate summary statistics
        SUMMARY[k, :] = [
            int(len(h['time_diff'])),
            np.median(h['time_diff']) * 1000,
            np.mean(h['time_diff']) * 1000,
            np.std(h['time_diff']) * 1000,
            np.min(h['time_diff']) * 1000,
            np.max(h['time_diff']) * 1000
        ]

    # Assign column and row names to SUMMARY matrix
    SUMMARY = pd.DataFrame(SUMMARY, index=instr, columns=['N', 'Md', 'M', 'SD', 'Min', 'Max'])
    SUMMARY['N'] = SUMMARY['N'].astype(int)

    # Plotting (if required)
    if plot:

        # Number of subplots
        num_plots = len(instr)

        # Calculate number of rows and columns for subplots
        ncols = 2
        nrows = (num_plots + ncols - 1) // ncols

        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3 * nrows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Define bin edges to align with the x-ticks
        
        # Generate custom bin edges
        binw_ms = binw * 1000
        bins = [filter_lower * 1000]
        start = (filter_lower * 1000) + binw_ms/2  # Start in milliseconds
        while start < filter_upper * 1000:
            bins.append(start)
            start += binw_ms
        # The last bin edge should be at the upper filter bound
        bins.append(filter_upper * 1000)

        # Plotting histograms
        for i, instrument in enumerate(instr):
            ax = axes[i]
            ax.grid(True, zorder=0)  # Ensure gridlines are behind the bars
            
            # Using custom bin edges
            sns.histplot(data=HIS[HIS['Instrument'] == instrument], x='d', bins=bins, kde=False, fill=True, color='white', edgecolor='black', ax=ax, zorder=2)
            
            # For Auto Binning
            #sns.histplot(data=HIS[HIS['Instrument'] == instrument], x='d', binwidth=binw_ms, kde=False, fill=True, color='white', edgecolor='black', ax=ax, zorder=2)
            
            ax.set_xlim(filter_lower * 1000, filter_upper * 1000)
            ax.set_xlabel('Onset time difference in ms')
            ax.set_ylabel('Count')
            ax.set_title(f'{instrument}')

        # Remove any unused subplots
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle('Histograms of Onset Time Differences')
        plt.tight_layout()
        plt.show()

    return SUMMARY

def summarise_sync(df=None):

    #This function summarises the calculated asynchronies.
    
    d_in_ms = df['asynch'] * 1000  # Convert to ms
    
    asynch = {
        'Pairwise asynchronization': np.std(d_in_ms),
        'Mean absolute asynchrony': np.mean(np.abs(d_in_ms)),
        'Mean pairwise asynchrony': np.mean(d_in_ms)
    }
    
    # Convert dictionary to a DataFrame with the proper orientation
    asynch_df = pd.DataFrame([asynch])
    return asynch_df


def bonferroni_adjust(p_values):
    """Manually apply Bonferroni adjustment to a list of p-values."""
    p_values = np.array(p_values)  # Ensure input is a NumPy array for element-wise operations
    num_tests = np.sum(~np.isnan(p_values))  # Count only non-NaN p-values for adjustment
    
    # Apply Bonferroni adjustment
    adjusted = np.where(~np.isnan(p_values), np.minimum(1, p_values * num_tests), np.nan)
    return adjusted


def summarise_sync_by_pair(pair_dict, bybeat=False, adjust=True):

    #Calculate summary statistics for asynchronies by instrument pairs
    
    if bybeat == False:
        # Reshape data
        m = pair_dict['asynch'].melt(var_name='Instrument', value_name='ms')
        m['ms'] *= 1000  # Convert to milliseconds

        # Unique instrument names
        instruments = m['Instrument'].unique()
        T = {'instr_pair' : [], 'tval': [], 'pval': []}

        # Perform t-tests for each instrument
        for inst in instruments:
            T['instr_pair'].append(inst)
            
            tmp = m[m['Instrument'] == inst]['ms']

            if tmp.isna().all():
                T['tval'].append(np.nan)
                T['pval'].append(np.nan)
            else:
                t_stat, p_value = ttest_1samp(tmp.dropna(), popmean=0)
                
                t_stat = np.round(t_stat,5)
                T['tval'].append(t_stat)
                
                p_value = np.round(p_value,5)
                T['pval'].append(p_value)
                

        # Calculate mean and standard deviation for each instrument pair
        M = pair_dict['asynch'].mean()
        M = np.round(M.values * 1000,5)
        
        SD = pair_dict['asynch'].std()
        SD = np.round(SD.values * 1000,5)

        # Combine results into a DataFrame
        T2 = pd.DataFrame({
            'N': pair_dict['asynch'].count(),
            'M': M,
            'SD': SD,
            'T': T['tval'],
            'pval': T['pval']
        })

        # Adjust p-values
        if adjust == True and not T2['pval'].isna().all():
            T2['pval'] = bonferroni_adjust(T2['pval'])

        # Prettify p-values
        T2['pval'] = T2['pval'].apply(lambda p: f'<0.001' if p < 0.001 else f'>0.999' if p > 0.999 else round(p, 3))
        
        return T2
    
    else:
        # Reshape data
        m = pair_dict['asynch'].melt(var_name='Instrument', value_name='ms')
        m['ms'] *= 1000  # Convert to milliseconds
        
        b = pair_dict['beatL'].melt(var_name='Instrument', value_name='Subdivision')
        m['beatL'] = b['Subdivision'].astype('category')  # Assign beat labels
        
        subdivisions = sorted(b['Subdivision'].unique())
        T = []

        # Loop through each subdivision (group by beat level)
        for subdiv in subdivisions:
            tmp = m[m['beatL'] == subdiv]
            n = tmp.shape[0]  # Sample size
            
            if n > 1:
                mean_asynch = tmp['ms'].mean()  # Mean
                std_asynch = tmp['ms'].std()   # Standard deviation
                
                # Perform one-sample t-test
                t_stat, p_value = ttest_1samp(tmp['ms'].dropna(), popmean=0)
            else:
                mean_asynch = np.nan
                std_asynch = np.nan
                t_stat = np.nan
                p_value = np.nan
            
            # Collect results
            T.append({
                'Subdivision': subdiv,
                'N': n,
                'M': round(mean_asynch,5),
                'SD': round(std_asynch,5),
                'T': round(t_stat,5),
                'pval': round(p_value,5)
            })
        
        # Create a DataFrame from results
        T2 = pd.DataFrame(T)

        # Adjust p-values using Bonferroni method
        if adjust == True and not T2['pval'].isna().all():
            T2['pval'] = bonferroni_adjust(T2['pval'])
        
        # Prettify p-values
        T2['pval'] = T2['pval'].apply(lambda p: f'<0.001' if p < 0.001 else f'>0.999' if p > 0.999 else round(p, 3))
        
        return T2


