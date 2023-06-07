import pandas as pd

def get_average_of_5_below(df):
    """
    A function that takes in a dataframe and a date and adds columns future_meal, future_insulin, future_exercise, and
    future_carbs to the dataframe. Each future column is the average of the next 5 measurements of that column after the
    date
    :param df: The basic dataframe
    :return: new dataframe with the new columns
    """

    df['future_meal'] = 0
    df['future_insulin'] = 0
    df['future_exercise'] = 0
    df['future_carbs'] = 0
    for i in range(len(df)):
        if i % 10000 == 0:
            print(i)
        if i + 24 < len(df):
            df.loc[i, 'future_meal'] = df.loc[i + 1:i + 24, 'mealsize 1'].sum()
            df.loc[i, 'future_insulin'] = df.loc[i + 1:i + 24, 'insulin 1'].sum()
            df.loc[i, 'future_exercise'] = df.loc[i + 1:i + 24, 'exercise 1'].sum()
            df.loc[i, 'future_carbs'] = df.loc[i + 1:i + 24, 'carbs 1'].sum()
        else:
            df.loc[i, 'future_meal'] = -1
            df.loc[i, 'future_insulin'] = -1
            df.loc[i, 'future_exercise'] = -1
            df.loc[i, 'future_carbs'] = -1
    df = df[df['future_meal'] != -1]
    df = df[df['future_insulin'] != -1]
    df = df[df['future_exercise'] != -1]
    df = df[df['future_carbs'] != -1]
    return df

if __name__ == '__main__':
    df_basic = pd.read_csv('basic_0.csv')
    print(len(df_basic))
    print(df_basic.head())
    df_self_alt = get_average_of_5_below(df_basic)
    print(df_self_alt.head())
    print(len(df_self_alt))
    #save the dataframe to a csv file
    df_self_alt.to_csv('self_sup_alt.csv', index=False)
    print(df_self_alt.head())
    print(df_self_alt['future_meal'].mean())
    print(df_self_alt['future_insulin'].mean())
    print(df_self_alt['future_exercise'].mean())
    print(df_self_alt['future_carbs'].mean())