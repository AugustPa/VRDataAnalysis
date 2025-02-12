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
    --start_trim          Time in seconds to trim from the beginning of each segment (default: 0.2 seconds)
    --end_trim            Time in seconds to trim from the end of each segment (default: 0.2 seconds)
    --min_segment_duration  Minimum duration (in seconds) for segments to be kept (default: 0, i.e. no filtering)
    --export_plotly       If set, generate an interactive Plotly visualization (and export as HTML)
    --show_end_markers    If set, show end markers on trajectories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import click
import os


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


def trim_segment_edges(df, start_trim_time=0.2, end_trim_time=0.2):
    """
    Trim the beginning and end of each trajectory segment to remove edge artifacts.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with a 'Segment' column.
      start_trim_time (float): Time (in seconds) to trim from segment start.
      end_trim_time (float): Time (in seconds) to trim from segment end.
      
    Returns:
      pd.DataFrame: DataFrame with trimmed segments.
    """
    def trim_group(group):
        start_time = group['Current Time'].min()
        end_time = group['Current Time'].max()
        mask = (group['Current Time'] >= start_time + pd.Timedelta(seconds=start_trim_time)) & \
               (group['Current Time'] <= end_time - pd.Timedelta(seconds=end_trim_time))
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


def plot_with_matplotlib(df, show_end_markers=True, output_path=None, reset_time=True):
    """
    Generate a static visualization of trajectory segments using Matplotlib.
    Each segment is plotted with a distinct color. Start and end markers are added.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with a 'Segment' column.
      show_end_markers (bool): Whether to show end markers.
      output_path (str): Base path for saving output files (without extension).
      reset_time (bool): Whether to reset time coloring for each segment.
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
        
        # Get step start time for continuous coloring
        step_start_time = step_data['Current Time'].min()
        
        for i, segment in enumerate(segments):
            seg_data = step_data[step_data['Segment'] == segment]
            
            # Calculate time for coloring
            if reset_time:
                # Reset time for each segment
                time_values = (seg_data['Current Time'] - seg_data['Current Time'].min()).dt.total_seconds()
                max_time = time_values.max()
                color_values = time_values / max_time if max_time > 0 else np.zeros_like(time_values)
            else:
                # Continuous time across segments
                time_values = (seg_data['Current Time'] - step_start_time).dt.total_seconds()
                max_time = (step_data['Current Time'] - step_start_time).dt.total_seconds().max()
                color_values = time_values / max_time if max_time > 0 else np.zeros_like(time_values)
            
            # Plot trajectory with time-based coloring
            points = axes[step_idx].scatter(
                seg_data['GameObjectPosX'], seg_data['GameObjectPosZ'],
                c=time_values, cmap='viridis', s=20
            )
            
            # Add markers for start (circle)
            axes[step_idx].plot(seg_data['GameObjectPosX'].iloc[0], seg_data['GameObjectPosZ'].iloc[0],
                     marker='o', color='black', markersize=10)
            # Add end marker (X) if enabled
            if show_end_markers:
                axes[step_idx].plot(seg_data['GameObjectPosX'].iloc[-1], seg_data['GameObjectPosZ'].iloc[-1],
                         marker='X', color='red', markersize=10)
        
        # Add colorbar for time
        plt.colorbar(points, ax=axes[step_idx], label='Time (s)')
        
        # Only show x label on bottom subplot
        if step_idx == len(steps) - 1:
            axes[step_idx].set_xlabel('GameObjectPosX')
        axes[step_idx].set_ylabel('GameObjectPosZ')
        axes[step_idx].set_title(f'Step {step} Trajectories')
        axes[step_idx].set_aspect('equal', adjustable='box')
        axes[step_idx].grid(True, linestyle='--', alpha=0.7)
    
    # Set the same limits for all subplots (only need to do once due to sharing)
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if output_path:
        # Save as both PDF and PNG
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{output_path}.png", bbox_inches='tight', dpi=300)
        click.echo(f"Matplotlib plots exported to {output_path}.pdf and {output_path}.png")
    
    plt.show()




def plot_with_plotly(df, show_end_markers=True, output_path=None, reset_time=True, n_frames=50):
    """
    Generate an interactive Plotly visualization of trajectory segments using
    vectorized subset updates. Each segment is a separate trace, and a slider
    controls the timeline. The function also:
      - Sets all subplots to share synchronized x/y axes with proper padding.
      - Defaults the view to the end of the timeline (complete trajectories).
      - Preserves the color information (based on time) in each animated trace.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with a 'Segment' column.
      show_end_markers (bool): Whether to show end markers.
      output_path (str): Base path for saving the output HTML file.
      reset_time (bool): If True, reset the time for each segment; otherwise use global time.
      n_frames (int): Number of frames for the slider animation.
    """
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import click

    # Determine steps (if available) to assign segments to subplots.
    if 'CurrentStep' in df.columns:
        steps = sorted(df['CurrentStep'].unique())
    else:
        steps = [0]
    
    # Map each step to a subplot (arranged in 2 columns)
    step_to_subplot = {}
    n_steps = len(steps)
    n_rows = (n_steps + 1) // 2  # Ceiling division for 2 columns
    for i, step in enumerate(steps):
        row = (i // 2) + 1
        col = (i % 2) + 1
        step_to_subplot[step] = (row, col)
    
    # Create subplots if more than one step; otherwise, use a single plot.
    if n_steps > 1:
        subplot_titles = [f'Step {step}' for step in steps]
        fig = make_subplots(rows=n_rows, cols=2,
                            subplot_titles=subplot_titles,
                            shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=0.1, horizontal_spacing=0.1)
    else:
        fig = go.Figure()

    # Precompute data for each segment.
    segments_data = []
    if 'CurrentStep' in df.columns:
        for step in steps:
            step_data = df[df['CurrentStep'] == step]
            for segment in sorted(step_data['Segment'].unique()):
                seg_df = step_data[step_data['Segment'] == segment]
                x_arr = seg_df['GameObjectPosX'].to_numpy()
                y_arr = seg_df['GameObjectPosZ'].to_numpy()
                # Determine time values (reset per segment or continuous across the step)
                if reset_time:
                    t_arr = (seg_df['Current Time'] - seg_df['Current Time'].min()).dt.total_seconds().to_numpy()
                else:
                    t_arr = (seg_df['Current Time'] - step_data['Current Time'].min()).dt.total_seconds().to_numpy()
                segments_data.append({
                    'step': step,
                    'segment': segment,
                    'x': x_arr,
                    'y': y_arr,
                    't': t_arr,
                    'subplot': step_to_subplot[step]
                })
    else:
        # Fallback: one step only.
        for segment in sorted(df['Segment'].unique()):
            seg_df = df[df['Segment'] == segment]
            x_arr = seg_df['GameObjectPosX'].to_numpy()
            y_arr = seg_df['GameObjectPosZ'].to_numpy()
            if reset_time:
                t_arr = (seg_df['Current Time'] - seg_df['Current Time'].min()).dt.total_seconds().to_numpy()
            else:
                t_arr = (seg_df['Current Time'] - df['Current Time'].min()).dt.total_seconds().to_numpy()
            segments_data.append({
                'step': 0,
                'segment': segment,
                'x': x_arr,
                'y': y_arr,
                't': t_arr,
                'subplot': (1, 1)
            })

    # Compute global max time (for color scaling) and global x/y limits.
    global_max_time = max([seg['t'].max() for seg in segments_data if len(seg['t']) > 0])
    x_min = min([np.min(seg['x']) for seg in segments_data if len(seg['x']) > 0])
    x_max = max([np.max(seg['x']) for seg in segments_data if len(seg['x']) > 0])
    y_min = min([np.min(seg['y']) for seg in segments_data if len(seg['y']) > 0])
    y_max = max([np.max(seg['y']) for seg in segments_data if len(seg['y']) > 0])
    # Add padding to the limits.
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]

    # Create base traces (one per segment) with empty data but proper marker setup.
    for seg in segments_data:
        row, col = seg['subplot']
        trace = go.Scatter(
            x=[], 
            y=[],
            mode='lines+markers',
            marker=dict(
                size=5,
                color=[],  # This will be updated in each frame.
                colorscale='Viridis',
                cmin=0,
                cmax=global_max_time
            ),
            line=dict(color='lightgrey', width=1),
            name=f"Step {seg['step']} - Segment {seg['segment']}"
        )
        if n_steps > 1:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)

    # Optionally, add static start (and end) markers that remain constant.
    for seg in segments_data:
        row, col = seg['subplot']
        # Start marker (green star)
        start_marker = go.Scatter(
            x=[seg['x'][0]],
            y=[seg['y'][0]],
            mode='markers',
            marker=dict(size=10, symbol='star', color='green'),
            name=f"Start {seg['segment']}",
            showlegend=False
        )
        if n_steps > 1:
            fig.add_trace(start_marker, row=row, col=col)
        else:
            fig.add_trace(start_marker)
        # End marker (red star)
        if show_end_markers:
            end_marker = go.Scatter(
                x=[seg['x'][-1]],
                y=[seg['y'][-1]],
                mode='markers',
                marker=dict(size=10, symbol='star', color='red'),
                name=f"End {seg['segment']}",
                showlegend=False
            )
            if n_steps > 1:
                fig.add_trace(end_marker, row=row, col=col)
            else:
                fig.add_trace(end_marker)

    # Build frames: each frame updates each segment trace using vectorized slicing.
    time_points = np.linspace(0, global_max_time, n_frames)
    frames = []
    for t_val in time_points:
        frame_traces = []
        for seg in segments_data:
            mask = seg['t'] <= t_val
            new_x = seg['x'][mask]
            new_y = seg['y'][mask]
            new_color = seg['t'][mask]
            frame_traces.append(dict(
                x=new_x,
                y=new_y,
                marker=dict(color=new_color)
            ))
        frames.append(go.Frame(data=frame_traces, name=f"frame_{t_val:.2f}"))
    fig.frames = frames

    # Build slider steps.
    slider_steps = []
    for t_val in time_points:
        slider_steps.append({
            "args": [[f"frame_{t_val:.2f}"],
                     {"frame": {"duration": 0, "redraw": False},
                      "mode": "immediate",
                      "transition": {"duration": 0}}],
            "label": f"{t_val:.1f}",
            "method": "animate"
        })

    # Set slider to default to the last frame (complete trajectory).
    slider_config = {
        "active": n_frames - 1,
        "currentvalue": {
            "font": {"size": 16},
            "prefix": "Time: ",
            "visible": True,
            "xanchor": "right"
        },
        "pad": {"b": 10, "t": 50},
        "steps": slider_steps
    }

    # Configure play/pause buttons and add the slider.
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top"
        }],
        sliders=[slider_config],
        title="Interactive Trajectories (Vectorized Subset Updates)",
        hovermode="closest"
    )

    # Synchronize the x and y axes across subplots by setting 'matches'
    fig.update_xaxes(range=x_range, matches='x')
    fig.update_yaxes(range=y_range, matches='y')

    # Set the default visible data to the final frame (complete trajectory).
    final_frame = frames[-1]
    num_animated_traces = len(segments_data)
    for i in range(num_animated_traces):
        fig.data[i].x = final_frame.data[i]['x']
        fig.data[i].y = final_frame.data[i]['y']
        fig.data[i].marker.color = final_frame.data[i]['marker']['color']

    # Optionally export the figure as HTML.
    if output_path:
        html_path = f"{output_path}.html"
        fig.write_html(html_path)
        click.echo(f"Plotly visualization exported to {html_path}")
    
    fig.show()






@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--velocity_threshold', default=100, help='Velocity threshold for jump detection.')
@click.option('--time_buffer', default=0.1, help='Time buffer (seconds) for filtering around jumps.')
@click.option('--start_trim', default=0, help='Time in seconds to trim from the beginning of each segment (default: 0.2 seconds).')
@click.option('--end_trim', default=0.2, help='Time in seconds to trim from the end of each segment (default: 0.2 seconds).')
@click.option('--min_segment_duration', default=0, help='Minimum segment duration (seconds) to keep (default: 0, i.e. no filtering).')
@click.option('--export_plotly', is_flag=True, help='Export interactive Plotly plot as an HTML file and display it.')
@click.option('--decimate_factor', default=1, help='Factor to reduce number of plotted points (e.g., 2 means plot every 2nd point).')
@click.option('--show_end_markers', is_flag=True, default=False, help='Show end markers on trajectories.')
@click.option('--reset_time', is_flag=True, default=True, help='Reset time coloring for each segment.')
def main(file_path, velocity_threshold, time_buffer, start_trim, end_trim, min_segment_duration, export_plotly, decimate_factor, show_end_markers, reset_time):
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
    df = trim_segment_edges(df, start_trim_time=start_trim, end_trim_time=end_trim)
    
    click.echo("Filtering jumps...")
    df = filter_jumps(df, velocity_threshold, time_buffer)
    
    if min_segment_duration > 0:
        click.echo("Filtering segments by duration...")
        df = filter_by_duration(df, min_segment_duration)
    
    # Apply decimation
    if decimate_factor > 1:
        click.echo(f"Decimating data by factor of {decimate_factor}...")
        df = df.iloc[::decimate_factor].copy()
    
    # Get base output path from input file path (remove extension)
    output_base = os.path.splitext(file_path)[0]
    
    if export_plotly:
        click.echo("Generating interactive Plotly visualization...")
        plot_with_plotly(df, show_end_markers=show_end_markers, output_path=output_base, reset_time=reset_time)
    else:
        click.echo("Generating static Matplotlib visualization...")
        plot_with_matplotlib(df, show_end_markers=show_end_markers, output_path=output_base, reset_time=reset_time)


if __name__ == "__main__":
    main()