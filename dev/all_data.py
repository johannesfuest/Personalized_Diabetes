# Script to combine all 30 patient dataframes
import pandas as pd

for i in range(1, 31):
    df = pd.read_csv(f'df_final_{i}.csv')
    if i == 1:
        all_data = df
        all_data['DeidentID'] = 1
    else:
        df['DeidentID'] = i
        all_data = pd.concat([all_data, df], axis=0)
    print(f'Patient {i} done!')

all_data.to_csv('all_data.csv', index=False)