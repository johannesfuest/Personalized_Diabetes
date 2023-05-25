import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('basic_5.csv')
print(len(df))
# sort df by CGM column
df = df.sort_values('insulin 1')
for i in range(1, 289):
    df = df[df[f'insulin {i}'] < 10000]
    if i % 10 == 0:
        print(f'insulin {i} done')
print(df.tail(100))
print(len(df))
df = df.sort_values('LocalDtTm')
df.to_csv('basic_5.csv', index=False)
