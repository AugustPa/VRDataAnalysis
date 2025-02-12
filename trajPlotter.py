#!/usr/bin/env python3
"""
VR Trajectory Data Analysis and Visualization Tool

This tool loads Unity-generated VR trajectory data from a CSV file, preprocesses it,
segments trajectories based on trial/step resets, trims edge artifacts, filters out
anomalous jumps, and displays the results using both static (Matplotlib) and interactive
(Plotly) visualizations.

Usage:
    python vr_trajectory_tool.py <file_path> [OPTIONS]

Options:
    --velocity_threshold  Velocity threshold for jump detection (default: 100)
    --time_buffer         Time buffer (in seconds) for filtering around jumps (default: 0.1)
    --edge_trim           Time in seconds to trim from the beginning and end of each segment (default: 0.2)
    --min_segment_duration  Minimum duration (in seconds) for segments to be kept (default: 0, i.e. no filtering)
    --export_plotly       If set, generate an interactive Plotly visualization (and export as HTML)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import click


def load_and_preprocess(file_path):
    """
    Load the CSV file and preprocess the data:
      - Convert 'Current Time' to datetime.
      - Compute velocity using the differences in GameObjectPosX and GameObjectPosZ.
    
    Parameters:
      file_path (str): Path to the CSV file.
      
    Returns:
      pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    # Convert "Current Time" to datetime
    df['Current Time'] = pd.to_datetime(df['Current Time'])
    
    # Compute velocity using vectorized operations
    dx = df['GameObjectPosX'].diff()
    dz = df['GameObjectPosZ'].diff()
    dt = df['Current Time'].diff().dt.total_seconds()
    # Avoid division by zero (or NaN for first row)
    dt = dt.replace(0, np.nan).fillna(0.01)
    df['Velocity'] = np.sqrt(dx**2 + dz**2) / dt
    
    return df


def segment_trajectories(df):
    """
    Segment the trajectory into discrete segments.
    A new segment is defined by a change in either 'CurrentTrial' or 'CurrentStep'.
    If these columns are not present, falls back to changes in 'GameObjectRotY'.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      
    Returns:
      pd.DataFrame: DataFrame with a new 'Segment' column.
    """
    if 'CurrentTrial' in df.columns and 'CurrentStep' in df.columns:
        # When either CurrentTrial or CurrentStep changes, mark a new segment.
        changes = df[['CurrentTrial', 'CurrentStep']].diff().fillna(0)
        segment_change = changes.ne(0).any(axis=1)
        df['Segment'] = segment_change.cumsum()
    else:
        # Fallback segmentation using GameObjectRotY changes.
        df['Segment'] = (df['GameObjectRotY'].diff().fillna(0) != 0).cumsum()
    return df


def trim_segment_edges(df, trim_time=0.2):
    """
    Trim the beginning and end of each trajectory segment to remove edge artifacts.
    By default, trims 200 ms from both the start and end of each segment.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with a 'Segment' column.
      trim_time (float): Time (in seconds) to trim from segment edges.
      
    Returns:
      pd.DataFrame: DataFrame with trimmed segments.
    """
    def trim_group(group):
        start_time = group['Current Time'].min()
        end_time = group['Current Time'].max()
        mask = (group['Current Time'] >= start_time + pd.Timedelta(seconds=trim_time)) & \
               (group['Current Time'] <= end_time - pd.Timedelta(seconds=trim_time))
        return group[mask]
    
    df_trimmed = df.groupby('Segment', group_keys=False).apply(trim_group)
    return df_trimmed


def filter_jumps(df, velocity_threshold, time_buffer):
    """
    Filter out data points associated with anomalous jumps.
    A jump is detected when the velocity exceeds the threshold.
    All data points within ±time_buffer seconds around the jump are removed.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      velocity_threshold (float): Velocity threshold to identify jumps.
      time_buffer (float): Time (in seconds) to buffer around jump events.
      
    Returns:
      pd.DataFrame: DataFrame after jump filtering.
    """
    df = df.copy()
    # Identify jumps where velocity exceeds threshold
    df['Jump'] = df['Velocity'] > velocity_threshold
    # Initialize removal mask as all False
    remove_mask = pd.Series(False, index=df.index)
    # Find indices of jump events
    jump_indices = df.index[df['Jump']]
    for jump_idx in jump_indices:
        jump_time = df.at[jump_idx, 'Current Time']
        lower_bound = jump_time - pd.Timedelta(seconds=time_buffer)
        upper_bound = jump_time + pd.Timedelta(seconds=time_buffer)
        remove_mask |= (df['Current Time'] >= lower_bound) & (df['Current Time'] <= upper_bound)
    # Filter out marked rows and drop the temporary column
    df_filtered = df[~remove_mask].copy()
    df_filtered.drop(columns=['Jump'], inplace=True)
    return df_filtered


def filter_by_duration(df, min_segment_duration):
    """
    Remove trajectory segments that are shorter than a minimum duration.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with a 'Segment' column.
      min_segment_duration (float): Minimum duration (in seconds) required for a segment.
      
    Returns:
      pd.DataFrame: Filtered DataFrame containing only segments with sufficient duration.
    """
    df = df.copy()
    df['Segment_Duration'] = df.groupby('Segment')['Current Time'].transform(
        lambda x: (x.max() - x.min()).total_seconds()
    )
    df_filtered = df[df['Segment_Duration'] >= min_segment_duration].copy()
    df_filtered.drop(columns=['Segment_Duration'], inplace=True)
    return df_filtered


def plot_with_matplotlib(df):
    """
    Generate a static visualization of trajectory segments using Matplotlib.
    Each segment is plotted with a distinct color. Start and end markers are added.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with a 'Segment' column.
    """
    # Get unique steps and trials
    steps = sorted(df['CurrentStep'].unique()) if 'CurrentStep' in df.columns else [0]
    trials = sorted(df['CurrentTrial'].unique()) if 'CurrentTrial' in df.columns else [0]
    
    # Create a subplot for each step, arranged vertically with shared axes
    n_steps = len(steps)
    fig, axes = plt.subplots(n_steps, 1, figsize=(12, 8 * n_steps), squeeze=False, sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Find global axis limits for synchronization
    x_min = df['GameObjectPosX'].min()
    x_max = df['GameObjectPosX'].max()
    y_min = df['GameObjectPosZ'].min()
    y_max = df['GameObjectPosZ'].max()
    
    # Add some padding to the limits
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    cmap = plt.get_cmap('viridis')
    
    for step_idx, step in enumerate(steps):
        step_data = df[df['CurrentStep'] == step] if 'CurrentStep' in df.columns else df
        segments = step_data['Segment'].unique()
        n_segments = len(segments)
        
        for i, segment in enumerate(segments):
            seg_data = step_data[step_data['Segment'] == segment]
            axes[step_idx].plot(
                seg_data['GameObjectPosX'], seg_data['GameObjectPosZ'],
                color=cmap(i / n_segments), label=f'Segment {segment}'
            )
            # Add markers for start (circle) and end (X)
            axes[step_idx].plot(seg_data['GameObjectPosX'].iloc[0], seg_data['GameObjectPosZ'].iloc[0],
                     marker='o', color='black')
            axes[step_idx].plot(seg_data['GameObjectPosX'].iloc[-1], seg_data['GameObjectPosZ'].iloc[-1],
                     marker='X', color='red')
        
        # Only show x label on bottom subplot
        if step_idx == len(steps) - 1:
            axes[step_idx].set_xlabel('GameObjectPosX')
        axes[step_idx].set_ylabel('GameObjectPosZ')
        axes[step_idx].set_title(f'Step {step} Trajectories')
        axes[step_idx].legend()
        axes[step_idx].set_aspect('equal', adjustable='box')
        axes[step_idx].grid(True, linestyle='--', alpha=0.7)
    
    # Set the same limits for all subplots (only need to do once due to sharing)
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()


def plot_with_plotly(df):
    """
    Generate an interactive Plotly visualization of trajectory segments.
    Each segment is a separate trace, with markers colored by time (seconds since segment start).
    Start and end markers are also displayed.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with a 'Segment' column.
    """
    # Get unique steps
    steps = sorted(df['CurrentStep'].unique()) if 'CurrentStep' in df.columns else [0]
    
    # Create subplots - one row for each step with shared axes
    fig = make_subplots(
        rows=len(steps), cols=1,
        subplot_titles=[f'Step {step}' for step in steps],
        vertical_spacing=0.05,
        shared_xaxes='all',  # Changed to 'all' to ensure complete sharing
        shared_yaxes='all'   # Changed to 'all' to ensure complete sharing
    )
    
    # Find global axis limits for synchronization
    x_min = df['GameObjectPosX'].min()
    x_max = df['GameObjectPosX'].max()
    y_min = df['GameObjectPosZ'].min()
    y_max = df['GameObjectPosZ'].max()
    
    # Add some padding to the limits
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Ensure square aspect ratio by making ranges equal if needed
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        center = (y_max + y_min) / 2
        y_min = center - x_range/2
        y_max = center + x_range/2
    else:
        center = (x_max + x_min) / 2
        x_min = center - y_range/2
        x_max = center + y_range/2
    
    cmap = plt.get_cmap('viridis')
    
    for step_idx, step in enumerate(steps):
        step_data = df[df['CurrentStep'] == step] if 'CurrentStep' in df.columns else df
        segments = np.sort(step_data['Segment'].unique())
        n_segments = len(segments)
        
        for i, segment in enumerate(segments):
            seg_df = step_data[step_data['Segment'] == segment]
            time_since_start = (seg_df['Current Time'] - seg_df['Current Time'].min()).dt.total_seconds()
            
            fig.add_trace(
                go.Scatter(
                    x=seg_df['GameObjectPosX'],
                    y=seg_df['GameObjectPosZ'],
                    mode='lines+markers',
                    marker=dict(
                        size=5,
                        color=time_since_start,
                        colorscale='Viridis',
                        colorbar=dict(title='Time (s)') if i == n_segments - 1 and step_idx == len(steps) - 1 else None,
                    ),
                    line=dict(color='lightgrey', width=1),
                    name=f'Step {step} - Segment {segment}',
                    text=[f"{t:.2f}s" for t in time_since_start]
                ),
                row=step_idx + 1, col=1
            )
            
            # Start marker (green star)
            fig.add_trace(
                go.Scatter(
                    x=[seg_df['GameObjectPosX'].iloc[0]],
                    y=[seg_df['GameObjectPosZ'].iloc[0]],
                    mode='markers',
                    marker=dict(size=10, symbol='star', color='green'),
                    name=f'Step {step} - Segment {segment} Start',
                    showlegend=False
                ),
                row=step_idx + 1, col=1
            )
            
            # End marker (red star)
            fig.add_trace(
                go.Scatter(
                    x=[seg_df['GameObjectPosX'].iloc[-1]],
                    y=[seg_df['GameObjectPosZ'].iloc[-1]],
                    mode='markers',
                    marker=dict(size=10, symbol='star', color='red'),
                    name=f'Step {step} - Segment {segment} End',
                    showlegend=False
                ),
                row=step_idx + 1, col=1
            )
    
    # Update all axes to maintain aspect ratio and sharing
    for i in range(len(steps)):
        fig.update_xaxes(
            range=[x_min, x_max],
            constrain='domain',
            scaleanchor=f'y{i+1}',
            scaleratio=1,
            showgrid=True,
            gridcolor='lightgrey',
            row=i+1,
            col=1
        )
        fig.update_yaxes(
            range=[y_min, y_max],
            constrain='domain',
            showgrid=True,
            gridcolor='lightgrey',
            row=i+1,
            col=1
        )
    
    # Only show x-axis title on bottom plot
    fig.update_xaxes(title_text='GameObjectPosX', row=len(steps), col=1)
    # Show y-axis title on all plots
    for i in range(len(steps)):
        fig.update_yaxes(title_text='GameObjectPosZ', row=i+1, col=1)
    
    fig.update_layout(
        title='Interactive Trajectories (Plotly)',
        hovermode='closest',
        height=400 * len(steps),  # Adjust height based on number of steps
        width=800,  # Fixed width
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    fig.write_html('interactive_trajectories.html')
    click.echo("Plotly HTML exported to interactive_trajectories.html")
    fig.show()


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--velocity_threshold', default=100, help='Velocity threshold for jump detection.')
@click.option('--time_buffer', default=0.1, help='Time buffer (seconds) for filtering around jumps.')
@click.option('--edge_trim', default=0.2, help='Time in seconds to trim from segment edges (default: 0.2 seconds).')
@click.option('--min_segment_duration', default=0, help='Minimum segment duration (seconds) to keep (default: 0, i.e. no filtering).')
@click.option('--export_plotly', is_flag=True, help='Export interactive Plotly plot as an HTML file and display it.')
@click.option('--decimate_factor', default=1, help='Factor to reduce number of plotted points (e.g., 2 means plot every 2nd point).')
def main(file_path, velocity_threshold, time_buffer, edge_trim, min_segment_duration, export_plotly, decimate_factor):
    """
    Process VR trajectory data from a CSV file by:
      - Preprocessing and velocity computation.
      - Segmenting based on trial/step resets.
      - Trimming edge artifacts.
      - Filtering out anomalous jumps.
      - Optionally filtering segments by minimum duration.
      - Visualizing the resulting trajectories.
    """
    click.echo("Loading and preprocessing data...")
    df = load_and_preprocess(file_path)
    
    click.echo("Segmenting trajectories...")
    df = segment_trajectories(df)
    
    click.echo("Trimming segment edges...")
    df = trim_segment_edges(df, trim_time=edge_trim)
    
    click.echo("Filtering jumps...")
    df = filter_jumps(df, velocity_threshold, time_buffer)
    
    if min_segment_duration > 0:
        click.echo("Filtering segments by duration...")
        df = filter_by_duration(df, min_segment_duration)
    
    # Apply decimation
    if decimate_factor > 1:
        click.echo(f"Decimating data by factor of {decimate_factor}...")
        df = df.iloc[::decimate_factor].copy()
    
    if export_plotly:
        click.echo("Generating interactive Plotly visualization...")
        plot_with_plotly(df)
    else:
        click.echo("Generating static Matplotlib visualization...")
        plot_with_matplotlib(df)


if __name__ == "__main__":
    main()