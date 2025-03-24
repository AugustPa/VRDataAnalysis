import os
import numpy as np
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and perform initial processing.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Current Time']).sort_values('Current Time')
        df['SourceFile'] = os.path.basename(file_path)  # Include the filename
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def process_dataframe(df: pd.DataFrame, trim_seconds: float = 1.0) -> pd.DataFrame:
    """
    Process the dataframe: trim the first and last second of each trial step and remove zero positions.
    """
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

def get_combined_df(directory: str, trim_seconds: float = 1.0) -> pd.DataFrame:
    """
    Recursively load and process CSV files from subdirectories and combine into a single DataFrame,
    incorporating metadata from the JSON file in each directory.
    """
    subdirectories = [
        os.path.join(directory, d) 
        for d in os.listdir(directory) 
        if os.path.isdir(os.path.join(directory, d))
    ]

    if not subdirectories:
        print(f"No subdirectories found in directory: {directory}")
        return pd.DataFrame()
    
    combined_dfs = []
    for subdir in subdirectories:
        subfolder_name = os.path.basename(subdir)
        print(f"Processing subfolder: {subfolder_name}")

        # 1. Identify the metadata JSON in the current folder
        metadata_files = glob.glob(os.path.join(subdir, '*_FlyMetaData.json'))
        metadata = None
        if metadata_files:
            # If there are multiple, just pick the first or handle accordingly
            meta_path = metadata_files[0]
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            print(f"No JSON metadata file found in subfolder: {subdir}")

        # 2. Parse the relevant fields from the metadata
        experimenter_name = None
        comments = None
        vr_fly_map = {}  # Will map "VR1" -> "FlyID"

        if metadata:
            experimenter_name = metadata.get("ExperimenterName", "")
            comments = metadata.get("Comments", "")
            flies_info = metadata.get("Flies", [])  # list of dicts

            # Build a dict for VR -> FlyID mapping
            for dct in flies_info:
                # dct might look like {"VR": "VR1", "FlyID": "41", ...}
                vr = dct.get("VR")
                fly_id = dct.get("FlyID")
                if vr and fly_id:
                    vr_fly_map[vr] = fly_id

        # 3. Load all CSVs in subfolder
        file_paths = [
            os.path.join(subdir, f) 
            for f in os.listdir(subdir) 
            if f.endswith('.csv')
        ]
        if not file_paths:
            print(f"No CSV files found in subfolder: {subdir}")
            continue

        dfs = []
        for csv_path in file_paths:
            df_loaded = load_csv(csv_path)
            df_processed = process_dataframe(df_loaded, trim_seconds)
            if df_processed.empty:
                print(f"No data loaded from {csv_path}")
                continue

            # 4. Add metadata columns if available
            if metadata:
                # Add the same experimenter name and comments to every row
                df_processed["ExperimenterName"] = experimenter_name
                df_processed["Comments"] = comments

                # Map VR -> FlyID
                # (If a particular VR doesn't exist in vr_fly_map, result will be NaN)
                df_processed["FlyID"] = df_processed["VR"].map(vr_fly_map)
            
            dfs.append(df_processed)

        if not dfs:
            print(f"No data frames were loaded for subfolder: {subfolder_name}")
            continue

        # 5. Concatenate all CSVs from this subfolder
        combined_df_subfolder = pd.concat(dfs, ignore_index=True)
        combined_dfs.append(combined_df_subfolder)
    
    # 6. Combine data from all subfolders
    if combined_dfs:
        final_df = pd.concat(combined_dfs, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

def add_trial_id_and_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a unique trial ID and compute step-by-step displacement as well as total displacement.
    """
    if df.empty:
        return df

    df['UniqueTrialID'] = df.groupby(['SourceFile', 'CurrentStep', 'CurrentTrial']).ngroup()
    df = df.sort_values(by=['UniqueTrialID', 'Current Time'])

    # Calculate stepwise displacement
    df['delta_x'] = df.groupby('UniqueTrialID')['GameObjectPosX'].diff()
    df['delta_y'] = df.groupby('UniqueTrialID')['GameObjectPosY'].diff()
    df['delta_z'] = df.groupby('UniqueTrialID')['GameObjectPosZ'].diff()

    df['step_distance'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2 + df['delta_z']**2)
    df['step_distance'] = df['step_distance'].fillna(0)

    # Total displacement per UniqueTrialID
    total_displacement = df.groupby('UniqueTrialID')['step_distance'].sum().reset_index()
    total_displacement.rename(columns={'step_distance': 'TotalDisplacement'}, inplace=True)
    df = df.merge(total_displacement, on='UniqueTrialID', how='left')

    return df

def classify_trials_by_displacement(df: pd.DataFrame, min_disp=0, max_disp=50):
    """
    Classify trials into stationary, normal, and excessive based on displacement thresholds.
    Returns three DataFrames and corresponding trial IDs.
    """
    if df.empty:
        return df, df, df, [], [], []

    total_displacement = df.groupby('UniqueTrialID')['TotalDisplacement'].first().reset_index()

    stationary_trial_ids = total_displacement[total_displacement['TotalDisplacement'] < min_disp]['UniqueTrialID'].unique()
    normal_trial_ids = total_displacement[
        (total_displacement['TotalDisplacement'] >= min_disp) &
        (total_displacement['TotalDisplacement'] <= max_disp)
    ]['UniqueTrialID'].unique()
    excessive_trial_ids = total_displacement[total_displacement['TotalDisplacement'] > max_disp]['UniqueTrialID'].unique()

    df_stationary = df[df['UniqueTrialID'].isin(stationary_trial_ids)].reset_index(drop=True)
    df_normal = df[df['UniqueTrialID'].isin(normal_trial_ids)].reset_index(drop=True)
    df_excessive = df[df['UniqueTrialID'].isin(excessive_trial_ids)].reset_index(drop=True)

    return df_stationary, df_normal, df_excessive, stationary_trial_ids, normal_trial_ids, excessive_trial_ids

def add_trial_time(df_normal: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'trial_time' column representing time since start of each trial.
    """
    if df_normal.empty:
        return df_normal
    df_normal['trial_time'] = df_normal.groupby('UniqueTrialID')['elapsed_time'].transform(lambda x: x - x.min())
    return df_normal

def plot_trajectories(df_group: pd.DataFrame, group_name: str, sample_size=None):
    """
    Plot trajectories for a given group of trials.
    """
    plt.figure(figsize=(10, 8))
    unique_trials = df_group['UniqueTrialID'].unique()

    # Optionally sample trials if there are too many
    if sample_size and len(unique_trials) > sample_size:
        np.random.seed(42)
        unique_trials = np.random.choice(unique_trials, size=sample_size, replace=False)

    for trial_id in unique_trials:
        trial_data = df_group[df_group['UniqueTrialID'] == trial_id]
        plt.plot(trial_data['GameObjectPosX'], trial_data['GameObjectPosZ'], alpha=0.5)

    plt.axis('equal')
    plt.xlabel('GameObjectPosX')
    plt.ylabel('GameObjectPosZ')
    plt.title(f'Trajectories of {group_name} Trials')
    plt.show()

def plot_displacement_distribution(df_normal: pd.DataFrame):
    """
    Plot distribution of maximum total displacement per trial for each source file as a boxplot.
    """
    if df_normal.empty:
        print("No data provided to plot displacement distribution.")
        return

    max_displacement_per_trial = df_normal.groupby(['SourceFile', 'UniqueTrialID'])['TotalDisplacement'].max().reset_index()
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=max_displacement_per_trial, x='SourceFile', y='TotalDisplacement')
    plt.xticks(rotation=90)
    plt.xlabel('Source File')
    plt.ylabel('Max Total Displacement per Trial (cm)')
    plt.title('Distribution of Max Total Displacement per Trial for Each Source File')
    plt.grid(True)
    plt.show()

def plot_displacement_by_step(df_normal: pd.DataFrame):
    """
    Plot average total displacement by CurrentStep.
    """
    if df_normal.empty:
        print("No data provided to plot displacement by step.")
        return

    displacement_per_step = df_normal.groupby('CurrentStep')['TotalDisplacement'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=displacement_per_step, x='CurrentStep', y='TotalDisplacement')
    plt.xlabel('Current Step')
    plt.ylabel('Average Total Displacement (cm)')
    plt.title('Average Total Displacement by Current Step')
    plt.grid(True)
    plt.show()

def plot_violin_displacement_by_step(df_normal: pd.DataFrame):
    """
    Plot distribution of total displacement by CurrentStep using a violin plot.
    """
    if df_normal.empty:
        print("No data provided for violin plot.")
        return

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_normal, x='CurrentStep', y='TotalDisplacement', scale='width', inner='quartile')
    plt.xlabel('Current Step')
    plt.ylabel('Total Displacement (cm)')
    plt.title('Distribution of Total Displacement by Current Step')
    plt.grid(True)
    plt.show()

def plot_directions_by_step(df: pd.DataFrame):
    """
    Plot histograms of final directional angles for each CurrentStep.
    """
    if df.empty:
        print("No data provided to plot directions by step.")
        return

    last_positions = df.groupby('UniqueTrialID').last().reset_index()
    last_positions['Angle'] = np.degrees(np.arctan2(last_positions['GameObjectPosZ'], last_positions['GameObjectPosX']))
    last_positions['Angle'] = last_positions['Angle'].apply(lambda x: x + 360 if x < 0 else x)

    unique_steps = sorted(last_positions['CurrentStep'].unique())
    plt.figure(figsize=(15, 10))
    for i, step in enumerate(unique_steps):
        plt.subplot(len(unique_steps)//2 + 1, 2, i + 1)
        subset = last_positions[last_positions['CurrentStep'] == step]
        plt.hist(subset['Angle'], bins=36, range=[0, 360], color='skyblue', edgecolor='black')
        plt.title(f'Current Step {step}')
        plt.xlabel('Directional Angle (degrees)')
        plt.ylabel('Frequency')
        plt.xlim([0, 360])
        plt.xticks(np.arange(0, 361, 45))
    plt.tight_layout()
    plt.show()

def plot_distance_distribution(df_normal: pd.DataFrame):
    """
    Plot distribution of final distances from origin for normal trials.
    """
    if df_normal.empty:
        print("No data provided to plot distance distribution.")
        return

    last_positions = df_normal.groupby('UniqueTrialID').last().reset_index()
    last_positions['DistanceFromOrigin'] = np.sqrt(last_positions['GameObjectPosX']**2 + last_positions['GameObjectPosZ']**2)
    plt.figure(figsize=(12, 6))
    sns.histplot(last_positions['DistanceFromOrigin'], bins=30, kde=True, color='blue')
    plt.xlabel('Distance from Origin (units)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Distances from Origin')
    plt.grid(True)
    plt.show()

def compare_two_steps(df_normal: pd.DataFrame, step1=0, step2=1):
    """
    Compare the final distances from origin between two sets of trials (defined by CurrentStep).
    Perform a Mann-Whitney U test and plot violin and density plots.
    """
    if df_normal.empty:
        print("No data to compare.")
        return

    df_normal['DistanceFromOrigin'] = np.sqrt(df_normal['GameObjectPosX']**2 + df_normal['GameObjectPosZ']**2)
    df_step1 = df_normal[df_normal['CurrentStep'] == step1]
    df_step2 = df_normal[df_normal['CurrentStep'] == step2]

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='CurrentStep', y='DistanceFromOrigin', data=df_normal)
    plt.title('Comparison of Final Distances from Origin by Step')
    plt.grid(True)
    plt.show()

    # Density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_step1['DistanceFromOrigin'], label=f'Step {step1}', fill=True)
    sns.kdeplot(df_step2['DistanceFromOrigin'], label=f'Step {step2}', fill=True)
    plt.title('Density Plot of Final Distances from Origin')
    plt.xlabel('Distance from Origin (units)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Statistical test
    stat, p = mannwhitneyu(df_step1['DistanceFromOrigin'], df_step2['DistanceFromOrigin'])
    print(f'Mann-Whitney U test results: U = {stat}, p-value = {p}')

def downsample_df(df: pd.DataFrame, factor: int = 6) -> pd.DataFrame:
    """
    Downsample the DataFrame by keeping every 'factor'-th row.
    """
    if df.empty:
        return df
    df['index_mod'] = df.index % factor
    df_reduced = df[df['index_mod'] == 0].copy()
    df_reduced.drop(columns=['index_mod'], inplace=True, errors='ignore')
    return df_reduced

def plot_single_frame(df_group: pd.DataFrame, trial_time_point: float, ax):
    """
    Plot trajectories up to a given trial_time_point on a provided Axes object.
    """
    unique_trials = df_group['UniqueTrialID'].unique()
    for trial_id in unique_trials:
        trial_data = df_group[(df_group['UniqueTrialID'] == trial_id) & (df_group['trial_time'] <= trial_time_point)]
        ax.plot(trial_data['GameObjectPosX'], trial_data['GameObjectPosZ'], alpha=0.2)

    ax.set_aspect('equal')
    ax.set_xlabel('GameObjectPosX')
    ax.set_ylabel('GameObjectPosZ')
    ax.set_title(f'Trajectories up to {trial_time_point} seconds')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])

def save_frame(df_group: pd.DataFrame, trial_time_point: float, frame_number: int, output_dir: str):
    """
    Generate and save a single frame of trajectory data.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_single_frame(df_group, trial_time_point, ax)
    filename = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
    plt.savefig(filename, dpi=100)
    plt.close(fig)

def generate_animation_frames(df_group: pd.DataFrame, output_dir: str, total_seconds: int = 20, fps: int = 10):
    """
    Generate a series of frames for animation, plotting trajectories up to each time point.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(total_seconds * fps + 1):
        trial_time_point = i / fps
        save_frame(df_group, trial_time_point, i, output_dir)
        if i < 5:  # print status for first few frames
            print(f"Frame {i} saved for time {trial_time_point:.2f}s")


def plot_trajectory_heatmap(df_group: pd.DataFrame, group_name: str, bins=50, exclude_radius=0.5):
    """
    Plot a heatmap of trajectory density for a given group of trials using a 2D histogram.
    Excludes all points within `exclude_radius` of the origin.
    """
    # Compute the distance of each point from the origin
    distances = (df_group['GameObjectPosX']**2 + df_group['GameObjectPosZ']**2)**0.5
    df_filtered = df_group[distances > exclude_radius]

    x = df_filtered['GameObjectPosX']
    z = df_filtered['GameObjectPosZ']

    plt.figure(figsize=(10, 8))
    counts, xedges, yedges, im = plt.hist2d(x, z, bins=bins, cmap='hot')
    plt.colorbar(im, label='Count')

    plt.xlabel('GameObjectPosX')
    plt.ylabel('GameObjectPosZ')
    plt.title(f'Trajectory Density Heatmap for {group_name} Trials (excluded radius={exclude_radius})')
    plt.axis('equal')
    plt.show()

def get_first_goal_reached(df_normal,
                           center_goal=(0, 60),
                           left_goal=(-10.416, 59.088),
                           right_goal=(10.416, 59.088),
                           threshold=3.5,
                           center_only_configs=None):
    """
    Given a dataframe of trial data, determine the first goal reached 
    and the time at which it was reached for each UniqueTrialID.

    Parameters
    ----------
    df_normal : pd.DataFrame
        The dataframe containing trial data. Must contain columns:
        ['UniqueTrialID', 'ConfigFile', 'trial_time', 'GameObjectPosX', 'GameObjectPosZ'].
    center_goal : tuple, optional
        Coordinates of the center goal.
    left_goal : tuple, optional
        Coordinates of the left goal.
    right_goal : tuple, optional
        Coordinates of the right goal.
    threshold : float, optional
        Distance threshold below which a goal is considered reached.
    center_only_configs : list of str, optional
        List of config file names that use only the center goal.

    Returns
    -------
    pd.DataFrame
        A dataframe with one row per UniqueTrialID, including:
        ['UniqueTrialID', 'ConfigFile', 'FirstReachedGoal', 'GoalReachedTime'].
    """
    
    # Default for center_only_configs if not provided
    if center_only_configs is None:
        center_only_configs = [
            "BinaryChoice10_BlackCylinder_control.json",
            "BinaryChoice10_constantSize_BlackCylinder_control.json"
        ]
    
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    results = []
    
    # Group by UniqueTrialID
    for trial_id, trial_data in df_normal.groupby('UniqueTrialID'):
        config = trial_data['ConfigFile'].iloc[0]
        
        # Determine relevant goals based on config
        if config in center_only_configs:
            goals = [('center', center_goal)]
        else:
            goals = [('left', left_goal), ('right', right_goal)]
        
        first_reached = None
        reached_time = None
        
        # Ensure data is time-sorted
        trial_data = trial_data.sort_values(by='trial_time')
        
        for idx, row in trial_data.iterrows():
            participant_pos = (row['GameObjectPosX'], row['GameObjectPosZ'])
            
            # Check each goal
            for goal_name, goal_pos in goals:
                dist = distance(participant_pos, goal_pos)
                if dist <= threshold:
                    first_reached = goal_name
                    reached_time = row['trial_time']
                    break
            
            if first_reached is not None:
                break
        
        # Collect results
        results.append((trial_id, config, first_reached, reached_time))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results, columns=['UniqueTrialID', 'ConfigFile', 'FirstReachedGoal', 'GoalReachedTime'])
    
    return results_df

def get_first_goal_reached_v2(df_normal,
                              center_goal=(0, 60),
                              left_goal=(-10.416, 59.088),
                              right_goal=(10.416, 59.088),
                              # Triple-goal positions (can be different from the above)
                              triple_center_goal=(0, 60),
                              triple_left_goal=(-20.5212, 56.381557),
                              triple_right_goal=(20.5212, 56.381557),
                              threshold=4,
                              center_only_configs=None,
                              triple_goals_configs=None):
    """
    Given a dataframe of trial data, determine the first goal reached 
    and the time at which it was reached for each UniqueTrialID.
    
    Allows for three categories of configs:
      1) center-only        -> uses center_goal
      2) left+right         -> uses left_goal, right_goal
      3) left+center+right  -> uses triple_left_goal, triple_center_goal, triple_right_goal

    Parameters
    ----------
    df_normal : pd.DataFrame
        The dataframe containing trial data. Must contain columns:
        ['UniqueTrialID', 'ConfigFile', 'trial_time', 'GameObjectPosX', 'GameObjectPosZ'].
    center_goal : tuple, optional
        Coordinates of the center goal for center-only configs.
    left_goal : tuple, optional
        Coordinates of the left goal (two-goal configs).
    right_goal : tuple, optional
        Coordinates of the right goal (two-goal configs).
    triple_center_goal : tuple, optional
        Coordinates of the center goal for three-goal configs.
    triple_left_goal : tuple, optional
        Coordinates of the left goal for three-goal configs.
    triple_right_goal : tuple, optional
        Coordinates of the right goal for three-goal configs.
    threshold : float, optional
        Distance threshold below which a goal is considered reached.
    center_only_configs : list of str, optional
        List of config file names that use only the center goal.
    triple_goals_configs : list of str, optional
        List of config file names that use three goals (left, center, right).

    Returns
    -------
    pd.DataFrame
        A dataframe with one row per UniqueTrialID, including:
        ['UniqueTrialID', 'ConfigFile', 'FirstReachedGoal', 'GoalReachedTime'].
    """
    
    # Default for center_only_configs if not provided
    if center_only_configs is None:
        center_only_configs = [
            "BinaryChoice10_BlackCylinder_control.json",
            "BinaryChoice10_constantSize_BlackCylinder_control.json"
        ]
    
    # Default for triple_goals_configs if not provided
    if triple_goals_configs is None:
        triple_goals_configs = [
            "3Cylinders111_constantSize.json"
        ]

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    results = []
    
    # Group by UniqueTrialID
    for trial_id, trial_data in df_normal.groupby('UniqueTrialID'):
        config = trial_data['ConfigFile'].iloc[0]
        
        # Determine relevant goals based on config
        if config in center_only_configs:
            # Only the center goal
            goals = [('center', center_goal)]
        elif config in triple_goals_configs:
            # Three goals, using the "triple" positions
            goals = [
                ('left', triple_left_goal),
                ('center', triple_center_goal),
                ('right', triple_right_goal)
            ]
        else:
            # Default case: left + right
            goals = [('left', left_goal), ('right', right_goal)]
        
        first_reached = None
        reached_time = None
        
        # Sort data by time within the trial
        trial_data = trial_data.sort_values(by='trial_time')
        
        # Check each position in chronological order
        for _, row in trial_data.iterrows():
            participant_pos = (row['GameObjectPosX'], row['GameObjectPosZ'])
            
            # Check each goal
            for goal_name, goal_pos in goals:
                dist = distance(participant_pos, goal_pos)
                if dist <= threshold:
                    first_reached = goal_name
                    reached_time = row['trial_time']
                    break
            
            if first_reached is not None:
                break
        
        # Collect results
        results.append((trial_id, config, first_reached, reached_time))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(
        results,
        columns=['UniqueTrialID', 'ConfigFile', 'FirstReachedGoal', 'GoalReachedTime']
    )
    
    return results_df
