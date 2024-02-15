import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import click

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df['Current Time'] = pd.to_datetime(df['Current Time'])
    df['Velocity'] = np.sqrt(df['GameObjectPosX'].diff()**2 + df['GameObjectPosZ'].diff()**2) / df['Current Time'].diff().dt.total_seconds().fillna(0.01)
    return df

def filter_jumps(df, velocity_threshold, time_buffer):
    df['High_Velocity'] = df['Velocity'] > velocity_threshold
    df_filtered = df[~df['High_Velocity']]
    return df_filtered

def segment_trajectories(df):
    df['Reset_Flag'] = ((df['GameObjectRotY'] == 0) & (df['GameObjectRotY'].shift(1) == 0)).astype(int)
    df['Segment'] = df['Reset_Flag'].cumsum()
    return df

def plot_trajectories(df, export_plotly=False):
    cmap = plt.get_cmap('viridis')
    segments = df['Segment'].unique()
    plt.figure(figsize=(12, 10))
    for i, segment in enumerate(segments):
        segment_df = df[df['Segment'] == segment]
        plt.plot(segment_df['GameObjectPosX'], segment_df['GameObjectPosZ'], color=cmap(i / len(segments)), marker='o', linestyle='-', markersize=2)
    plt.xlabel('GameObjectPosX')
    plt.ylabel('GameObjectPosZ')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    if export_plotly:
        fig = make_subplots()
        for i, segment in enumerate(segments):
            segment_df = df[df['Segment'] == segment]
            fig.add_trace(
                go.Scatter(x=segment_df['GameObjectPosX'], y=segment_df['GameObjectPosZ'], mode='lines+markers', name=f'Segment {segment}', marker=dict(color=cmap(i / len(segments)))))
        fig.update_layout(title='Interactive Trajectories Visualization', xaxis_title='GameObjectPosX', yaxis_title='GameObjectPosZ', showlegend=False)
        fig.write_html('interactive_trajectories.html')

@click.command()

@click.argument('file_path', type=click.Path(exists=True))#, help='Path to the CSV file containing trajectory data.')
# @click.option('--file_path', type=str, required=True, help='Path to the CSV file containing trajectory data.')
@click.option('--velocity_threshold', default=100, help='Velocity threshold to identify jumps.')
@click.option('--time_buffer', default=5, help='Time buffer for filtering jumps.')
@click.option('--export_plotly', is_flag=True, help='Export visualization as an interactive Plotly HTML file.')
def main(file_path, velocity_threshold, time_buffer, export_plotly):
    """
    Process trajectory data to visualize and analyze movements, removing jumps and segmenting by resets.
    """
    df = load_and_preprocess(file_path)
    df = filter_jumps(df, velocity_threshold, time_buffer)
    df = segment_trajectories(df)
    plot_trajectories(df, export_plotly)

if __name__ == "__main__":
    main()
