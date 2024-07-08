import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

