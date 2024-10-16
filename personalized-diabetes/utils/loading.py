import pandas as pd
import os

DATA_DIR = 'CTR3_tables'
DATA_FILE_EXTENSION = '.txt'

# set current working dir to dir of file
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_table(filename:str, date_cols=[], nrows=None):
    if not DATA_FILE_EXTENSION in filename: 
        filename+=DATA_FILE_EXTENSION

    df = pd.read_csv(
        filepath_or_buffer=os.path.join(DATA_DIR, filename),
        sep='|',
        true_values=['Yes'],
        false_values=['No'],
        nrows=nrows,
        date_format='%Y-%m-%d %H:%M:%S',
        parse_dates=date_cols
    )

    return df

