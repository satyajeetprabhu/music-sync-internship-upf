import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

def seconds_to_hms(x, pos):
    """Convert seconds to HH:MM:SS format."""
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    

def plot_by_beat(df=None, instr=None, beat=None, virtual=None, pcols=2, griddeviations=False, colourpalette='Set2', pointsize=1):
    DF = pd.DataFrame()
    S = pd.DataFrame()
    
    # Loop across instruments
    for k in range(len(instr)):
        tmp = df[[instr[k], beat, virtual]].copy()
        tmp.columns = ['instr', 'beat', 'virtual']
        IBI = tmp['virtual'].diff().median()
        tmp['VSD'] = tmp['instr'] - tmp['virtual']
        tmp['VSDR'] = tmp['VSD'] / (tmp['virtual'] - tmp['virtual'].shift(1, fill_value=IBI))
        tmp['name'] = instr[k]
        DF = pd.concat([DF, tmp])
        tmp['beatF'] = tmp['beat'].astype('category')

        s = tmp.groupby('beatF').agg(M=('VSDR', lambda x: x.mean() * 100)).reset_index()
        s['Time'] = tmp['instr'].min()
        s['name'] = instr[k]
        S = pd.concat([S, s])

    DF['name'] = DF['name'].astype('category')
    S['name'] = S['name'].astype('category')

    unique_beats = sorted(DF['beat'].unique())
    # Create a mapping from beat values to numbers
    beat_mapping = {beat: i + 1 for i, beat in enumerate(unique_beats)}
    # Assign the corresponding number to each beat value
    DF['beat_number'] = DF['beat'].map(beat_mapping)
    S['beat_number'] = S['beatF'].map(beat_mapping)

    # Y-axis limits
    y_max = df[instr].dropna().values.max()  # Drop np.inf and -np.inf values
    y_min = 0

    # Plot
    num_plots = len(instr)
    num_rows = num_plots // pcols + (num_plots % pcols > 0)
    fig, axes = plt.subplots(nrows=num_rows, ncols=pcols, figsize=(12, 4*num_rows), squeeze=True, sharex=True, sharey=True)
    
    # Check if axes is a single AxesSubplot or an array of them
    if num_plots == 1:
        axes = [axes]  # Convert single AxesSubplot to a list
    else:
        axes = axes.flatten()  # Flatten the array of AxesSubplot

    for ax, instrument in zip(axes, instr):
        data = DF[DF['name'] == instrument]
        sns.scatterplot(x=data['beat_number'] + data['VSDR'], y=data['instr'], hue=data['name'], ax=ax, palette=colourpalette, s=pointsize*12, alpha=0.9, legend=False)
        
        if griddeviations:
            s_data = S[S['name'] == instrument]
            
            for _, row in s_data.iterrows():
                # Check if 'M' is NaN
                if pd.isna(row['M']):
                    formatted_label = "-"
                else:
                    # Format 'M' to one decimal point followed by '%'
                    #formatted_label = f"{row['M']:.0f}%"
                    formatted_label = f"{round(row['M'])}%"
                
                ax.text(x=row['beat_number'], y=0, s=formatted_label, ha='center', va='center', fontsize=8, 
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        ax.set_title(f'{instrument}')
        ax.grid(True, which='major', linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', linestyle='-', linewidth=0.25)
        
        ax.xaxis.set_major_locator(MultipleLocator(1))  # Set major ticks for x-axis
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))  # Set minor ticks for x-axis
        ax.set_xlim(DF['beat_number'].min()-0.5, DF['beat_number'].max()+0.5)  # Limit x-ticks to min(beat)-0.5, max(beat)+0.5

        # Set the x-ticks to match the adjusted beat numbers (starting at 1)
        tick_positions = range(1, len(unique_beats) + 1)
        ax.set_xticks(tick_positions)
        # Set the x-tick labels to be the unique_beats array
        ax.set_xticklabels(unique_beats, y=-0.02)
        
        ax.yaxis.set_major_locator(MultipleLocator(60))  # Set major ticks for y-axis every minute
        ax.yaxis.set_minor_locator(MultipleLocator(30))  # Set minor ticks for y-axis every 30 seconds
        ax.yaxis.set_major_formatter(FuncFormatter(seconds_to_hms)) # Format the y-ticks to show HH:MM:SS
        ax.set_ylim(y_min, y_max + y_max * 0.05)  # Set y-limits with a small buffer
        ax.set_ylabel('')

    # Remove unused subplots if any
    for i in range(len(instr), len(axes)):
        fig.delaxes(axes[i])

    # Common labels
    fig.text(0.55, 0.04, f'Beat ({beat})', ha='center' , va='center', fontsize=14)
    fig.text(0.04, 0.5, 'Time (HH:MM:SS)', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()


def boxplot_by_beat(df=None, instr=None, beat=None, virtual=None, pcols=2, color='lightblue'):
    DF = pd.DataFrame()
    S = pd.DataFrame()
    
    # Loop across instruments
    for k in range(len(instr)):
        tmp = df[[instr[k], beat, virtual]].copy()
        tmp.columns = ['instr', 'beat', 'virtual']
        IBI = tmp['virtual'].diff().median()
        tmp['VSD'] = tmp['instr'] - tmp['virtual']
        tmp['VSDR'] = tmp['VSD'] / (tmp['virtual'] - tmp['virtual'].shift(1, fill_value=IBI))
        tmp['name'] = instr[k]
        DF = pd.concat([DF, tmp])
        tmp['beatF'] = tmp['beat'].astype('category')

        s = tmp.groupby('beatF').agg(M=('VSDR', lambda x: x.mean() * 100)).reset_index()
        s['Time'] = tmp['instr'].min()
        s['name'] = instr[k]
        S = pd.concat([S, s])

    DF['name'] = DF['name'].astype('category')
    S['name'] = S['name'].astype('category')

    unique_beats = sorted(DF['beat'].unique())
    # Create a mapping from beat values to numbers
    beat_mapping = {beat: i + 1 for i, beat in enumerate(unique_beats)}
    # Assign the corresponding number to each beat value
    DF['beat_number'] = DF['beat'].map(beat_mapping)
    S['beat_number'] = S['beatF'].map(beat_mapping)

    # Plot
    num_plots = len(instr)
    num_rows = num_plots // pcols + (num_plots % pcols > 0)
    fig, axes = plt.subplots(nrows=num_rows, ncols=pcols, figsize=(10, 3*num_rows), squeeze=True, sharex=True, sharey=True)
    
    # Check if axes is a single AxesSubplot or an array of them
    if num_plots == 1:
        axes = [axes]  # Convert single AxesSubplot to a list
    else:
        axes = axes.flatten()  # Flatten the array of AxesSubplot

    for ax, instrument in zip(axes, instr):
        
        data = DF[DF['name'] == instrument]
        sns.boxplot(data=data, x='beat_number', y=DF['VSD'] * 1000, ax=ax, color=color, width=0.5, fliersize=2)

        ax.set_title(f'{instrument}', fontsize=14)
        ax.grid(True, linestyle='-', linewidth=0.5)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=12)

    # Remove unused subplots if any
    for i in range(len(instr), len(axes)):
        fig.delaxes(axes[i])

    # Common labels    
    fig.text(0.5, 0.0, f'Beat ({beat})', ha='center' , va='center', fontsize=14)
    fig.text(0.0, 0.5, 'Asynchrony (ms)', ha='center', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout()
    plt.show()


def plot_by_pair(df=None, bybeat=False, reference=0, colourpalette='Pastel2'):
    # Ensure required columns exist in the dataframe
    if 'asynch' not in df or ('beatL' not in df and bybeat):
        raise ValueError("Dataframe must contain 'asynch' column and 'beatL' column if bybeat=True")
    
    # Convert asynch to long format
    m = pd.melt(df['asynch'], var_name='Instrument', value_name='ms')
    m['ms'] = m['ms'] * 1000  # Convert to milliseconds
    m['Instrument'] = m['Instrument'].astype('category')
    
    if not bybeat:  # Asynchronies
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Horizontal violin plot without grey edge line
        sns.violinplot(
            data=m,
            y='Instrument',
            x='ms',
            hue=None,
            ax=ax,
            scale='width',
            palette=colourpalette,
            alpha=0.5,
            orient='h',
            linewidth=1,  # Violin plot edge line width
            linecolor='black',  
            inner=None  # Removes iterquantile range bar, {“box”, “quart”, “point”, “stick”, None}
        )
        
        # Jitter plot
        sns.stripplot(
            data=m,
            y='Instrument',
            x='ms',
            color='grey',
            size=2,
            ax=ax,
            jitter=0.15,
            alpha=1,
            zorder=2
        )

        # Mean summary line
        means = m.groupby('Instrument')['ms'].mean()
        for i, mean in enumerate(means):
            ax.plot([mean, mean], [i - 0.2, i + 0.2], color='black', linewidth=2, zorder=3)

        # Reference line
        ax.axvline(reference, color='orange', linestyle='dashed')
        
        # Grid lines
        ax.set_axisbelow(False)
        ax.grid(visible=True, which='both', axis='both', linestyle='-', linewidth=0.2, alpha=0.7, color = 'black')

        # Set labels
        ax.set_ylabel('Instrument pairs', fontsize=14)
        ax.set_xlabel('Synchrony (ms)', fontsize=14)
        #ax.set_title('Synchrony by Instrument Pair')

        plt.show()

    if bybeat:  # Synchronies by beat
        b = pd.melt(df['beatL'], var_name='variable', value_name='value')
        m['beatL'] = b['value'].astype('category')
        
        # Ensure color palette has enough colors for beat levels
        colpal = sns.color_palette(colourpalette, len(m['beatL'].unique()))
    
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Boxplot with unique colors per beat level
        sns.boxplot(
            data=m,
            x='Instrument',
            y='ms',
            hue='beatL',
            ax=ax,
            palette=colpal,
            showfliers=False,
            linewidth=0.5,
            dodge=True
        )
        
        # Reference line
        ax.axhline(reference, color='orange', linestyle='dashed')

        # Grid lines
        ax.grid(visible=True, which='both', axis='both', linestyle='-', linewidth=0.1, alpha=0.7, color = 'black')
        
        # Set labels
        ax.set_xlabel('Instrument pairs', fontsize=14)
        ax.set_ylabel('Synchrony (ms)', fontsize=14)
        #ax.set_title('Synchrony by Instrument Pair and Beat')

        # Move the legend box outside the plot area
        ax.legend(title='Subdivision', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

        plt.show()

    #return fig


