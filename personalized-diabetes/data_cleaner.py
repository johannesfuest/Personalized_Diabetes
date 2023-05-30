import pandas as pd

if __name__ == '__main__':
    for i in range(0,1):
        print('Reading in df for patient {}'.format(i))
        df = pd.read_csv(f'basic_{i}.csv')
        print('Dropping rows with CGM > 1000')
        df = df[df['CGM'] < 1000]
        print('Dropping rows with CGM < 0')
        df = df[df['CGM'] >= 0]
        print('Dropping rows damaged insulin values')
        df = df[df[[f'insulin {j}' for j in range(1,289)]].min(axis=1) >= 0]
        df = df[df[[f'insulin {j}' for j in range(1,289)]].max(axis=1)  < 1000]
        print('Dropping rows with damaged carb values')
        df = df[df[[f'carbs {j}' for j in range(1,289)]].min(axis=1) >= 0]
        df = df[df[[f'carbs {j}' for j in range(1,289)]].max(axis=1) < 1000]
        print('Dropping rows with damaged exercise values')
        df = df[df[[f'exercise {j}' for j in range(1,289)]].min(axis=1) > -0.01]
        df = df[df[[f'exercise {j}' for j in range(1,289)]].max(axis=1) < 100]
        print('Dropping rows with damaged mealsize values')
        df = df[df[[f'mealsize {j}' for j in range(1,289)]].min(axis=1) > -0.01]
        df = df[df[[f'mealsize {j}' for j in range(1,289)]].max(axis=1) < 1000]
        df.to_csv(f'basic_{i}_cleaned.csv', index=False)
        print(f'Data cleaning for patient {i} done!')
