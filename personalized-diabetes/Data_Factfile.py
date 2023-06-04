import pandas as pd
import numpy as np

#  Quick script to establish some baseline properties of the data
if __name__== '__main__':
    df_results = pd.DataFrame()
    # give df_results the empy columns patient, n_rows, avg_cgm, avg_insulin, avg_exercise, avg_mealsize, avg_glucose
    df_results['patient'] = []
    df_results['n_rows'] = []
    df_results['avg_cgm'] = []
    df_results['avg_insulin'] = []
    df = pd.read_csv('basic_0.csv')
    for i in range(1, 31):
        rowlist = []
        rowlist.append(i)
        df_patient = df[df['DeidentID'] == i]
        rowlist.append(len(df_patient))
        rowlist.append(df_patient['CGM'].mean())
        ins = 0
        meal = 0
        carbs = 0
        exercise = 0
        for j in range(1, 289):
            ins += df_patient[f'insulin {j}'].mean()
            meal += df_patient[f'mealsize {j}'].mean()
            carbs += df_patient[f'carbs {j}'].mean()
            exercise += df_patient[f'exercise {j}'].mean()
        rowlist.append(ins)
        rowlist.append(exercise)
        rowlist.append(meal)
        rowlist.append(carbs)
        df_patient.sort_values(by=['LocalDtTm'], inplace=True)
        print(df_patient.head())
        df_patient_train = df_patient.iloc[:int(len(df_patient) * 0.8)]
        df_patient_test = df_patient.iloc[int(len(df_patient) * 0.8):]
        rowlist.append(df_patient_train['CGM'].mean())
        rowlist.append(df_patient_test['CGM'].mean())
        ins_train = 0
        meal_train = 0
        carbs_train = 0
        exercise_train = 0
        ins_test = 0
        meal_test = 0
        carbs_test = 0
        exercise_test = 0
        for j in range(1, 289):
            ins_train += df_patient_train[f'insulin {j}'].mean()
            meal_train += df_patient_train[f'mealsize {j}'].mean()
            carbs_train += df_patient_train[f'carbs {j}'].mean()
            exercise_train += df_patient_train[f'exercise {j}'].mean()
            ins_test += df_patient_test[f'insulin {j}'].mean()
            meal_test += df_patient_test[f'mealsize {j}'].mean()
            carbs_test += df_patient_test[f'carbs {j}'].mean()
            exercise_test += df_patient_test[f'exercise {j}'].mean()
        rowlist.append(ins_train)
        rowlist.append(ins_test)
        rowlist.append(exercise_train)
        rowlist.append(exercise_test)
        rowlist.append(meal_train)
        rowlist.append(meal_test)
        rowlist.append(carbs_train)
        rowlist.append(carbs_test)
        if i == 1:
            df_results = pd.DataFrame([rowlist],
                                      columns=["patient", "nrwos", "avg_cgm", "avg_insulin", "avg_exercise",
                                               "avg_mealsize", "avg_glucose", "avg_glucose_train", "avg_glucose_test",
                                                  "avg_insulin_train", "avg_insulin_test", "avg_exercise_train",
                                                    "avg_exercise_test", "avg_mealsize_train", "avg_mealsize_test",
                                                        "avg_carbs_train", "avg_carbs_test"])
        else:
            df_results.loc[len(df_results)] = rowlist
    df_results.to_csv('data_summary.csv')