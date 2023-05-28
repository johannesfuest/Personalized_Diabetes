import pandas as pd

if __name__ == '__main__':
    for i in range(1, 31):
        print('Reading in df for patient {}'.format(i))
        df = pd.read_csv(f'basic_{i}.csv')
        print('Dropping rows with CGM > 1000')
        df = df[df['CGM'] < 1000]
        print('Dropping rows with CGM < 0')
        df = df[df['CGM'] > 0]
        print('Dropping rows damaged insulin values')
        df = df[df['insulin 1'] > 0]
        df = df[df['insulin 1'] < 1000]
        print('Dropping rows with damaged carb values')
        df = df[df['carbs 1'] > 0]
        df = df[df['carbs 1'] < 1000]
        print('Dropping rows with damaged exercise values')
        df = df[df['exercise 1'] > -0.01]
        df = df[df['exercise 1'] < 100]
        print('Dropping rows with damaged mealsize values')
        df = df[df['mealsize 1'] > -0.01]
        df = df[df['mealsize 1'] < 1000]
        df.to_csv(f'basic_{i}.csv', index=False)
        print(f'Data cleaning for patient {i} done!')
