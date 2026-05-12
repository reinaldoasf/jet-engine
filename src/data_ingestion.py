import pandas as pd
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_cmapss_train_data(train_file_path: str) -> pd.DataFrame:
    """
    Loads the C-MAPSS training dataset. 
    Since training data is 'run-to-failure', the RUL is simply the difference 
    between the engine's maximum recorded cycle and the current cycle.
    """
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    sensor_cols = [f'sensor_measurement_{i}' for i in range(1, 22)]
    columns.extend(sensor_cols)
    
    logging.info(f"Loading training data from {train_file_path}...")
    train_df = pd.read_csv(train_file_path, sep=r'\s+', header=None, names=columns)
    
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_in_cycles': 'max_cycle'}, inplace=True)
    
    train_df = pd.merge(train_df, max_cycles, on='unit_number')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycle'], inplace=True)
    
    logging.info("Training dataset loaded and RUL successfully calculated.")
    return train_df


def load_cmapss_test_data(test_file_path: str, rul_file_path: str) -> pd.DataFrame:
    """
    Loads the C-MAPSS test dataset and correctly maps the true RUL 
    from the label file to every cycle of each unit.
    """
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    sensor_cols = [f'sensor_measurement_{i}' for i in range(1, 22)]
    columns.extend(sensor_cols)
    
    logging.info(f"Loading test data from {test_file_path}...")
    test_df = pd.read_csv(test_file_path, sep=r'\s+', header=None, names=columns)
    
    logging.info(f"Loading true RUL labels from {rul_file_path}...")
    true_rul = pd.read_csv(rul_file_path, sep=r'\s+', header=None, names=['RUL_at_last_cycle'])
    true_rul['unit_number'] = true_rul.index + 1 
    
    max_cycles = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_in_cycles': 'max_cycle'}, inplace=True)
    
    rul_reference = pd.merge(max_cycles, true_rul, on='unit_number')
    test_df = pd.merge(test_df, rul_reference, on='unit_number', how='left')
    
    test_df['RUL'] = test_df['RUL_at_last_cycle'] + (test_df['max_cycle'] - test_df['time_in_cycles'])
    test_df.drop(columns=['max_cycle', 'RUL_at_last_cycle'], inplace=True)
    
    logging.info("Dataset loaded and RUL successfully mapped.")
    return test_df