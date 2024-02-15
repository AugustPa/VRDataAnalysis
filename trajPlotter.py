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


def segment_trajectories(df):
    df['Reset_Flag'] = ((df['GameObjectRotY'] == 0) & (df['GameObjectRotY'].shift(1) == 0)).astype(int)
    df['Segment'] = df['Reset_Flag'].cumsum()
    return df

# Erosion and Dilation: Identify and remove jumps, then filter segments by duration
def filter_jumps(df, velocity_threshold, time_buffer):
    df['Jump'] = df['Velocity'] > velocity_threshold
    jumps = df[df['Jump']].index

    # Initialize 'Remove' column as False for all rows before the loop
    df['Remove'] = False

    for jump in jumps:
        time_of_jump = df.loc[jump, 'Current Time']
        df.loc[(df['Current Time'] >= time_of_jump - pd.Timedelta(seconds=time_buffer)) & 
               (df['Current Time'] <= time_of_jump + pd.Timedelta(seconds=time_buffer)), 'Remove'] = True

    df_filtered = df[~df['Remove']]  # Directly use the 'Remove' column for filtering
    return df_filtered.drop(['Jump', 'Remove'], axis=1)

# Filtering segments based on minimum duration
def filter_by_duration(df, min_segment_duration):
    df['Segment_Duration'] = df.groupby('Segment')['Current Time'].transform(lambda x: (x.max() - x.min()).total_seconds())
    valid_segments = df.groupby('Segment').filter(lambda x: x['Segment_Duration'].iloc[0] >= min_segment_duration)
    return valid_segments.drop(['Segment_Duration'], axis=1)

# Plot with Matplotlib
def plot_with_matplotlib(df):
    segments = df['Segment'].unique()
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('viridis')
    
    for segment in segments:
        segment_data = df[df['Segment'] == segment]
        plt.plot(segment_data['GameObjectPosX'], segment_data['GameObjectPosZ'], color=cmap(segment / len(segments)))
    
    plt.xlabel('GameObjectPosX')
    plt.ylabel('GameObjectPosZ')
    plt.gca().set_aspect('equal', 'box')
    plt.show()

# Plot with Plotly
def plot_with_plotly(df):

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Creating an interactive Plotly figure with a slider for selecting trials
    fig = make_subplots()
    segments = df['Segment'].unique()
    cmap = plt.get_cmap('viridis')

    # Adding each segment as a trace to enable or disable viewing
    for i, segment in enumerate(segments): 
        segment_df = df[df['Segment'] == segment]

        # First, convert 'Current Time' to a numerical format (e.g., seconds since the start)
        # segment_df['Time_Since_Start'] = (segment_df['Current Time'] - segment_df['Current Time'].min()).dt.total_seconds()

        fig.add_trace(
            go.Scatter(
                x=segment_df['GameObjectPosX'],
                y=segment_df['GameObjectPosZ'],
                mode='lines+markers',
                marker=dict(
                    size=1, 
                    color=(segment_df['Current Time'] - segment_df['Current Time'].min()).dt.total_seconds(),
                    colorscale='Magma',  # Specify the color scale to use
                    # colorbar=dict(title='Time'),
                    showscale=False
                ),
                name=f'Trial {segment}'
            )
        )

        # # add a star or marker at the start of each segment, and another marker at the end
        # fig.add_trace(
        #     go.Scatter(
        #         x=[segment_df['GameObjectPosX'].iloc[0]],
        #         y=[segment_df['GameObjectPosZ'].iloc[0]],
        #         mode='markers',
        #         marker=dict(size=10, color='black'),
        #         showlegend=False
        #     )
        # )

        # fig.add_trace(
        #     go.Scatter(
        #         x=[segment_df['GameObjectPosX'].iloc[-1]],
        #         y=[segment_df['GameObjectPosZ'].iloc[-1]],
        #         mode='markers',
        #         marker=dict(size=10, color='black'),
        #         showlegend=False
        #     )
        # )


    #add a marker at 0,0
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(size=10, color='black'),
            showlegend=False
        )
    )

    # # Updating layout to include a slider for trial selection
    # fig.update_layout(
    #     title='Interactive Trajectories with Trial Selection',
    #     xaxis_title='GameObjectPosX',
    #     yaxis_title='GameObjectPosZ',
    #     legend_title='Trials',
    #     hovermode='closest',
    #     plot_bgcolor='white',
    #     showlegend=True,
    #     sliders=[{
    #         'currentvalue': {'prefix': 'Trial: '},
    #         'steps': [{'label': f'Trial {segment}', 'method': 'update', 'args': [{'visible': [s == segment for s in segments]}]} for segment in segments]
    #     }]
    # )

    # Ensure aspect ratio is equal
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Exporting the figure as HTML
    html_file_path = 'interactive_trajectories.html'
    fig.write_html(html_file_path)
    fig.show()

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
    # Example usage:
    df = load_and_preprocess(file_path)
    df = segment_trajectories(df)
    df = filter_jumps(df, velocity_threshold, time_buffer)
    # df = filter_by_duration(df, min_segment_duration=15)
    plot_with_matplotlib(df)
    if export_plotly:
        plot_with_plotly(df)  # Uncomment to use Plotly for plotting


if __name__ == "__main__":
    main()
