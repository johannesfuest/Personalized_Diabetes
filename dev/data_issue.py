import pandas as pd
import matplotlib.pyplot as plt
# df = pd.read_csv('df_final_5.csv')
# print(len(df))
# # sort df by CGM column
# df = df.sort_values('insulin 1')
# for i in range(1, 289):
#     df = df[df[f'insulin {i}'] < 10000]
#     if i % 10 == 0:
#         print(f'insulin {i} done')
# print(df.tail(100))
# print(len(df))
# df = df.sort_values('LocalDtTm')
# df.to_csv('df_final_5.csv', index=False)

df = pd.read_csv('df_final_26.csv')

print(len(df))
# sort df by CGM column
df = df.sort_values('carbs 1')
df = df[['carbs 1', 'LocalDtTm']]
#for i in range(1, 289):
#    df = df[df[f'insulin {i}'] < 10000]
#    if i % 10 == 0:
#        print(f'insulin {i} done')
print(df.tail(10))
print(len(df))
df = df.sort_values('LocalDtTm')
#df.to_csv('df_final_5.csv', index=False)

# plto carbs 1 over time
plt.plot(df['LocalDtTm'], df['carbs 1'])
plt.show()
