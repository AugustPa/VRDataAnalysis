import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from typing import List

def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file and perform initial processing."""
    try:
        df = pd.read_csv(file_path, parse_dates=['Current Time']).sort_values('Current Time')
        df['SourceFile'] = os.path.basename(file_path)  # Include the filename
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def process_dataframe(df: pd.DataFrame, trim_seconds: float = 1.0) -> pd.DataFrame:
    """Process the dataframe: trim first and last second of each trial step and remove zero positions."""
    if df.empty:
        return df
    df['elapsed_time'] = (df['Current Time'] - df['Current Time'].min()).dt.total_seconds()
    df['CurrentTrial'] = df['CurrentTrial'].astype(int)
    df['CurrentStep'] = df['CurrentStep'].astype(int)
    df['VR'] = df['VR'].astype(str)
    df = df.sort_values(['CurrentTrial', 'CurrentStep', 'elapsed_time'])
    grouped = df.groupby(['CurrentTrial', 'CurrentStep'])
    df = grouped.apply(lambda x: x[(x['elapsed_time'] >= trim_seconds) & 
                                   (x['elapsed_time'] <= x['elapsed_time'].max() - trim_seconds)]).reset_index(drop=True)
    # Remove rows where both positions are zero
    df = df[(df['GameObjectPosX'] != 0) | (df['GameObjectPosZ'] != 0)]
    return df

def load_and_process_data(file_paths: List[str], trim_seconds: float) -> List[pd.DataFrame]:
    """Load and process all data files in the list."""
    dfs = []
    for f in file_paths:
        df = process_dataframe(load_csv(f), trim_seconds)
        if not df.empty:
            dfs.append(df)
        else:
            print(f"No data loaded from {f}")
    return dfs

def create_subplot_layout(n_files: int) -> (int, int):
    """Calculate the number of rows and columns for the subplot layout."""
    n_cols = 2  # Fixed to 2 columns
    n_rows = (n_files + n_cols - 1) // n_cols
    return n_rows, n_cols


def add_traces_for_configfile(fig: go.Figure, config_data: pd.DataFrame, config_file: str,
                              row: int, col: int, vr_colors: dict, vr_legend: set) -> go.Figure:
    """Add traces for each VR, step, and trial for a specific ConfigFile to the figure."""
    trace_count = 0  # Initialize the trace count
    if not config_data.empty:
        for sourceFile in config_data['SourceFile'].unique():
            file_data = config_data[config_data['SourceFile'] == sourceFile]
            for vr in file_data['VR'].unique():
                vr_data = file_data[file_data['VR'] == vr]
                showlegend = vr not in vr_legend
                if showlegend:
                    vr_legend.add(vr)
                for trial in vr_data['CurrentTrial'].unique():
                    trial_data = vr_data[vr_data['CurrentTrial'] == trial]
                    for step in trial_data['CurrentStep'].unique():
                        step_data = trial_data[trial_data['CurrentStep'] == step]
                        step_data = step_data[(step_data['GameObjectPosX'] != 0) |
                                              (step_data['GameObjectPosZ'] != 0)]
                        if not step_data.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=step_data['GameObjectPosX'],
                                    y=step_data['GameObjectPosZ'],
                                    mode='lines',
                                    name=f"{vr} (File: {sourceFile})" if showlegend else None,
                                    line=dict(color=vr_colors.get(vr, 'black')),
                                    showlegend=showlegend
                                ),
                                row=row, col=col
                            )
                            showlegend = False
                            trace_count += 1
    print(f"Total traces drawn: {trace_count}")
    return fig


def update_axes(fig: go.Figure, row: int, col: int, min_val: float, max_val: float) -> go.Figure:
    """Update axes for a specific subplot, ensuring 1:1 scale and synchronization."""
    fig.update_xaxes(
        title_text='X Position',
        range=[min_val, max_val],
        constrain='domain',
        scaleanchor="y",
        scaleratio=1,
        matches='x',
        row=row, 
        col=col
    )
    fig.update_yaxes(
        title_text='Z Position',
        range=[min_val, max_val],
        constrain='domain',
        scaleanchor="x",
        scaleratio=1,
        matches='y',
        row=row, 
        col=col
    )
    return fig

def create_figure(df: pd.DataFrame, config_files: List[str], subfolder_name: str) -> go.Figure:
    if df.empty:
        print(f"No data to plot for subfolder {subfolder_name}.")
        return None
    n_rows, n_cols = create_subplot_layout(len(config_files))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=config_files,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.005
    )

    # Define colors for VRs
    vr_colors = {
        'VR1': 'blue',
        'VR2': 'green',
        'VR3': 'red',
        'VR4': 'orange'
    }

    vr_legend = set()

    for i, config_file in enumerate(config_files):
        config_data = df[df['ConfigFile'] == config_file]
        row, col = i // n_cols + 1, i % n_cols + 1
        fig = add_traces_for_configfile(fig, config_data, config_file, row, col, vr_colors, vr_legend)

    # Update axes ranges
    all_x = df['GameObjectPosX']
    all_y = df['GameObjectPosZ']
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    range_max = max(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    overall_min = x_center - range_max / 2
    overall_max = x_center + range_max / 2

    for i in range(len(config_files)):
        row, col = i // n_cols + 1, i % n_cols + 1
        fig = update_axes(fig, row, col, overall_min, overall_max)

    fig.update_layout(
        height=600*n_rows,
        width=600*n_cols,
        title_text=f"Ant Positions by ConfigFile - {subfolder_name}"
    )

    return fig

def plot_ant_positions(df: pd.DataFrame, config_files: List[str], output_file: str, subfolder_name: str):
    """Create and save the plot."""
    fig = create_figure(df, config_files, subfolder_name)
    if fig is None:
        return
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Plot saved as {output_file}")


def main(directory: str, trim_seconds: float = 1.0):
    # Get all subdirectories
    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) 
                      if os.path.isdir(os.path.join(directory, d))]

    if not subdirectories:
        print(f"No subdirectories found in directory: {directory}")
        return

    for subdir in subdirectories:
        subfolder_name = os.path.basename(subdir)
        print(f"Processing subfolder: {subfolder_name}")
        file_paths = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.csv')]

        if not file_paths:
            print(f"No CSV files found in subfolder: {subdir}")
            continue

        dfs = []
        for f in file_paths:
            df = process_dataframe(load_csv(f), trim_seconds)
            if not df.empty:
                dfs.append(df)
            else:
                print(f"No data loaded from {f}")

        if not dfs:
            print(f"No data frames were loaded for subfolder: {subfolder_name}")
            continue

        combined_df = pd.concat(dfs, ignore_index=True)
        unique_config_files = combined_df['ConfigFile'].unique()
        print(f"Unique ConfigFiles in {subfolder_name}: {unique_config_files}")

        output_file = os.path.join(subdir, f"{subfolder_name}_ant_positions.html")
        plot_ant_positions(combined_df, unique_config_files, output_file, subfolder_name)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide the directory path as an argument.")
