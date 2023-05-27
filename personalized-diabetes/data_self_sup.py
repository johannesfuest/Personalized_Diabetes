import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def get_self_sup_df(df):
    """
    A function that takes a dataframe and returns a dataframe with the next row as the targets.
    :param df: the dataframe to be transformed
    :return: a dataframe with the next row as target
    """
    # sort df by LocalDtTm
    df = df.sort_values('LocalDtTm')
    # loop over all columns and shift them by one
    for i in range(1, 289):
        df[f'insulin {i} target'] = df[f'insulin {i}'].shift(-1)
        df[f'mealsize {i} target'] = df[f'mealsize {i}'].shift(-1)
        df[f'carbs {i} target'] = df[f'carbs {i}'].shift(-1)
        df[f'exercise {i} target'] = df[f'exercise {i}'].shift(-1)
    #drop na values
    df = df.dropna()
    return df
def save_self_sup(df, patient):
    """
    A function that takes a basic patient dataframe and saves it as dataframe for self-supervised learning
    :param df: The dataframe
    :param patient: The patient number
    :return:
    """
    df = get_self_sup_df(df)
    df.to_csv(f'self_{patient}.csv', index=False)
def test_get_self_sup_df():
    # test get_self_sup_df
    for i in range(0, 1):
        df = pd.read_csv(os.path.join('..', 'personalized-diabetes', f'basic_{i}.csv'))
        len_pre = len(df)
        df = df.sort_values('LocalDtTm')
        df = get_self_sup_df(df)
        len_post = len(df)
        for i in range(100):
            assert df.iloc[i]['insulin 1 target'] == df.iloc[i + 1]['insulin 1']
            assert len_pre == len_post + 1
    print('test passed')
    return

if __name__ == '__main__':
    for i in range(1,31):
        df = pd.read_csv(os.path.join('..', 'personalized-diabetes', f'basic_{i}.csv'))
        save_self_sup(df, i)
        print(f'Patient {i} done!')