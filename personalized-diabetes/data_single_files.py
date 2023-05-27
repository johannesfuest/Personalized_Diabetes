# Script to combine all 30 patient dataframes
import pandas as pd


if __name__ == '__main__':
    for i in range(1, 31):
        df = pd.read_csv(f'basic_{i}.csv')
        df_self = pd.read_csv(f'self_{i}.csv')
        if i == 1:
            all_data = df
            all_data_self = df_self
            all_data['DeidentID'] = 1
            all_data_self['DeidentID'] = 1
        else:
            df['DeidentID'] = i
            df_self['DeidentID'] = i
            all_data = pd.concat([all_data, df], axis=0)
            all_data_self = pd.concat([all_data_self, df_self], axis=0)
        print(f'Patient {i} done!')
    all_data.to_csv('basic_0.csv', index=False)
    all_data_self.to_csv('self_0.csv', index=False)