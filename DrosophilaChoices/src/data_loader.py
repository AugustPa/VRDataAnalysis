import os
import pandas as pd
import json

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def list_run_folders(self):
        """ List all experiment run folders within the RunData directory. """
        run_folders = [os.path.join(self.base_path, d) for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        print(f"Found folders: {run_folders}")
        return run_folders

    def load_csv_data(self, folder_path):
        """ Load all CSV files from a given folder into a pandas DataFrame, 
            including folder name as an additional column. """
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"Loading CSV files from {folder_path}: {csv_files}")
        if not csv_files:
            print(f"No CSV files found in {folder_path}. Skipping...")
            return pd.DataFrame()  # Return an empty DataFrame if no CSV files found
        df_list = []
        folder_name = os.path.basename(folder_path)  # Get the folder name from the folder path
        for file_name in csv_files:
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            df['folder_name'] = folder_name  # Add folder name as a new column
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)


    def load_all_data(self):
        """ Load all CSV data from all folders and concatenate them into a single DataFrame. """
        all_data_frames = []
        for folder in self.list_run_folders():
            df = self.load_csv_data(folder)
            if df is not None:
                all_data_frames.append(df)
        if all_data_frames:
            return pd.concat(all_data_frames, ignore_index=True)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data


