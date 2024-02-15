import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import click

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--velocity_threshold', default=100, type=int, help='Velocity threshold for filtering. Default is 100.')
@click.option('--time_buffer', default=5, type=int, help='Time buffer for jump scrubbing in seconds. Default is 5.')
@click.option('--export_plotly', is_flag=True, help='Flag to enable Plotly HTML export.')
def process_trajectory(file_path, velocity_threshold, time_buffer, export_plotly):
    """
    Processes the trajectory data from a CSV file, filters out jumps, segments based on resets, 
    generates a log plot of velocities, and visualizes trajectories using Matplotlib and optionally Plotly.
    
    Args:
        file_path (str): Path to the input CSV file.
        velocity_threshold (int): Velocity threshold for filtering.
        time_buffer (int): Time buffer for jump scrubbing in seconds.
        export_plotly (bool): Enable Plotly HTML export.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    df['Current Time'] = pd.to_datetime(df['Current Time'])

    # Reset detection and splitting
    df['Reset_Flag'] = ((df['GameObjectRotY'] == 0) & (df['GameObjectRotY'].shift(1) == 0)).astype(int)
    df['Segment'] = df['Reset_Flag'].cumsum()

    # Calculate velocity
    df['Velocity'] = np.sqrt(df['GameObjectPosX'].diff()**2 + df['GameObjectPosZ'].diff()**2) / df['Current Time'].diff().dt.total_seconds().fillna(0.01)

    # Filter out jumps within each segment
    df_filtered = df[df['Velocity'] < velocity_threshold]

    # Generate log plot of velocities
    plt.figure(figsize=(10, 6))
    plt.hist(df_filtered['Velocity'], bins=100, log=True)
    plt.xlabel('Velocity (units/second)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Log Plot of Velocities')
    plt.show()

    # Plot trajectories using Matplotlib
    cmap = plt.get_cmap('viridis')
    segments = df_filtered['Segment'].unique()

    plt.figure(figsize=(12, 10))
    for i, segment in enumerate(segments):
        segment_df = df_filtered[df_filtered['Segment'] == segment]
        plt.plot(segment_df['GameObjectPosX'], segment_df['GameObjectPosZ'], color=cmap(i / len(segments)))

    plt.xlabel('GameObjectPosX')
    plt.ylabel('GameObjectPosZ')
    plt.title('Trajectories Visualization')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Optional Plotly HTML export
    if export_plotly:
        fig = make_subplots()
        for i, segment in enumerate(segments):
            segment_df = df_filtered[df_filtered['Segment'] == segment]
            fig.add_trace(
                go.Scatter(x=segment_df['GameObjectPosX'], y=segment_df['GameObjectPosZ'], mode='lines+markers', name=f'Segment {segment}'))

        fig.update_layout(title='Interactive Trajectories Visualization', xaxis_title='GameObjectPosX', yaxis_title='GameObjectPosZ')
        fig.write_html('interactive_trajectories.html')
        click.echo('Plotly visualization exported as interactive_trajectories.html.')

if __name__ == '__main__':
    process_trajectory()
