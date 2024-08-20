import pandas as pd
import numpy as np


def load_data(timestamp, directory_path, scene_name):
    file_paths = [directory_path + f'{timestamp}_{scene_name}_VR{i}_.csv' for i in range(1, 5)]
    dataframes = [pd.read_csv(fp) for fp in file_paths]
    for df in dataframes:
        df['trial_timestamp'] = timestamp
        df['Current Time'] = pd.to_datetime(df['Current Time'])
        # calculate distance of each step
        df['step_distance'] = np.sqrt((df['SensPosX'].diff())**2 + (df['SensPosY'].diff())**2)
        df['step_distance_mm'] = df['step_distance'] * 4.5
        #calculate time between each step with Current Time: Timestamp('2024-05-16 14:16:35.300000')
        df['time_diff'] = df['Current Time'].diff()
        df['time_diff_ms'] = df['time_diff'].dt.total_seconds() * 1000
        #calculate speed of each step
        df['speed_mm_s'] = df['step_distance_mm'] / df['time_diff_ms'] *1000

    return dataframes