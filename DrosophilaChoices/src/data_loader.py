import os
import pandas as pd
import json
import glob

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

    def load_json_data(self, folder_path):
        """ Load a JSON configuration file ending with 'sequenceConfig.json' and return it as a dictionary. """
        # Search for files that match the 'sequenceConfig.json' pattern
        json_files = glob.glob(os.path.join(folder_path, '*sequenceConfig.json'))
        if not json_files:
            print(f"No JSON config file ending with 'sequenceConfig.json' found in {folder_path}. Skipping...")
            return None
        json_path = json_files[0]  # Assuming only one match is relevant or taking the first match
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    
    def load_json_metadata(self, folder_path):
        """ Load a JSON metadata file ending with 'MetaData.json' and return it as a dictionary. """
        json_files = glob.glob(os.path.join(folder_path, '*MetaData.json'))
        if not json_files:
            print(f"No JSON metadata file ending with 'MetaData.json' found in {folder_path}. Skipping...")
            return None
        json_path = json_files[0]  # Assuming only one match is relevant or taking the first match
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        return json_data

    def enrich_dataframe(self, df, json_config_data, json_metadata_data):
        """ Enrich the DataFrame with JSON data based on the CurrentStep index and add metadata info. """
        if 'CurrentStep' in df.columns and json_config_data:
            # Map sequence configurations
            step_config = {idx: seq for idx, seq in enumerate(json_config_data['sequences'])}
            df['duration'] = df['CurrentStep'].map(lambda x: step_config.get(x, {}).get('duration'))
            df['configFile'] = df['CurrentStep'].map(lambda x: step_config.get(x, {}).get('parameters', {}).get('configFile'))
        
        if json_metadata_data:
            # Add experimenter name and comments to each row
            df['ExperimenterName'] = json_metadata_data.get('ExperimenterName', '').strip()
            df['Comments'] = json_metadata_data.get('Comments', '').strip()
        
        return df


    def load_all_data(self):
        """ Load all data from all folders, combining CSV and JSON data into a single DataFrame. """
        all_data_frames = []
        for folder in self.list_run_folders():
            df = self.load_csv_data(folder)
            json_data = self.load_json_data(folder)  # Load the JSON configuration
            json_metadata_data = self.load_json_metadata(folder)
            if df is not None:
                df = self.enrich_dataframe(df, json_data, json_metadata_data)  # Enrich DataFrame with JSON data
                all_data_frames.append(df)
        if all_data_frames:
            return pd.concat(all_data_frames, ignore_index=True)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data

