import pandas as pd
from tqdm.auto import tqdm
import sys

pd.set_option("display.max_rows", 4000)
pd.set_option("mode.chained_assignment", None)
sys.path.append("..")
tqdm.pandas()

from utils.loading import load_table

def get_24_hour_bins(df, date, df_name):
    """
    Given a DataFrame or Series indexed by datetime and a specified end 'date', 
    returns a reversed list of 288 values (5-min intervals over 24 hours) ending at 'date'.
    Missing intervals are filled with 0. Also supports different df_name which 
    indicates which column to use for data extraction if df is a DataFrame.

    If df is a Series, it is assumed to be the column of interest.
    """
    start_time = date - pd.Timedelta('1 day')
    intervals = pd.date_range(start_time, date, freq="5min")

    col_map = {
        "insulin": "DeliveredValue",
        "SMBG": "Carbs",
        "Meal": "MealSize",
        "System": "Exercising",
        "CGM": "CGM"
    }
    col = col_map[df_name]

    if isinstance(df, pd.DataFrame):
        # If DataFrame, ensure time index
        if 'LocalDtTm' in df.columns:
            df = df.set_index('LocalDtTm', drop=False).sort_index()
        series = df[col].reindex(intervals, fill_value=0)[::-1]
    else:
        # If Series
        df = df[~df.index.duplicated(keep='first')]
        series = df.reindex(intervals, fill_value=0)[::-1]

    return series.tolist()[:288]


def get_self_sup_df(df):
    """
    Adds four columns to the df for each row that are the average of the next 24 measurements of mealsize, insulin, exercise, and carbs.
    """
    df = df.copy()
    df["future_meal"] = 0.0
    df["future_insulin"] = 0.0
    df["future_exercise"] = 0.0
    df["future_carbs"] = 0.0
    
    # Sort the dataframe by time
    df = df.sort_values("LocalDtTm")
    df = df.reset_index(drop=True)
    
    for i in tqdm(range(len(df)), desc="Adding self-supervised columns"):   
        if i + 24 < len(df):
            df.loc[i, "future_meal"] = df.loc[i + 1 : i + 24, "mealsize_1"].sum()
            df.loc[i, "future_insulin"] = df.loc[i + 1 : i + 24, "insulin_1"].sum()
            df.loc[i, "future_exercise"] = df.loc[i + 1 : i + 24, "exercise_1"].sum()
            df.loc[i, "future_carbs"] = df.loc[i + 1 : i + 24, "carbs_1"].sum()
        else:
            df.loc[i, "future_meal"] = -1
            df.loc[i, "future_insulin"] = -1
            df.loc[i, "future_exercise"] = -1
            df.loc[i, "future_carbs"] = -1
    df = df[df["future_meal"] != -1]
    df = df[df["future_insulin"] != -1]
    df = df[df["future_exercise"] != -1]
    df = df[df["future_carbs"] != -1]
    return df.copy()


def combine_dfs(basics, selfs, test=False):
    """
    Concatenates all basic and self-supervised dataframes into two final CSV files.
    """
    all_data = []
    all_data_self = []
    for i, (basic_df, self_df) in enumerate(zip(basics, selfs)):
        # Patient numbering starts at 1
        pid = i + 1
        basic_df["DeidentID"] = pid
        self_df["DeidentID"] = pid
        all_data.append(basic_df)
        all_data_self.append(self_df)
        print(f"Patient {pid} done!")

    all_data = pd.concat(all_data, ignore_index=True)
    all_data_self = pd.concat(all_data_self, ignore_index=True)

    if test:
        all_data.to_csv("basic_0_test.csv", index=False)
        all_data_self.to_csv("self_0_test.csv", index=False)
    else:
        all_data.to_csv("basic_0.csv", index=False)
        all_data_self.to_csv("self_0.csv", index=False)


def preprocess_data():
    """
    Main function to:
    1. Load data tables.
    2. Filter relevant columns.
    3. Generate basic and self-supervised dataframes per patient.
    4. Compute past 288 5-min windows for CGM, insulin, carbs, mealsize, exercise.
    5. Count non-imputed CGM values and remove entries where count < 260.
    6. Combine and save results.
    """
    # Dictionary of tables and date columns
    tables = {
        "Meter": ["DeidentID", "DataDtTm"],
        "CGM": ["DeidentID", "DisplayTime"],
        "CGMCal": ["DeidentID", "DisplayTime"],
        "Ketone": ["DeidentID", "DataDtTm"],
        "MonitorBasalBolus": ["DeidentID", "LocalDeliveredDtTm"],
        "MonitorCGM": ["DeidentID", "LocalDtTmAdjusted"],
        "MonitorCorrectionBolus": ["DeidentID", "LocalDeliveredDtTm"],
        "MonitorMeal": ["DeidentID", "LocalDtTm"],
        "MonitorMealBolus": ["DeidentID", "LocalDeliveredDtTm"],
        "MonitorSMBG": ["DeidentID", "LocalDtTm"],
        "MonitorSystem": ["DeidentID", "LocalDtTm"],
        "MonitorTotalBolus": ["DeidentID", "LocalDeliveredDtTm"],
        "Pump": ["DeidentID", "DataDtTm"],
    }

    print("Reading in files from public study...")
    df_MonitorBasalBolus = load_table("MonitorBasalBolus", date_cols=tables["MonitorBasalBolus"])
    df_MonitorCGM = load_table("MonitorCGM", date_cols=tables["MonitorCGM"])
    df_MonitorCorrectionBolus = load_table("MonitorCorrectionBolus", date_cols=tables["MonitorCorrectionBolus"])
    df_MonitorMeal = load_table("MonitorMeal", date_cols=tables["MonitorMeal"])
    df_MonitorMealBolus = load_table("MonitorMealBolus", date_cols=tables["MonitorMealBolus"])
    df_MonitorSMBG = load_table("MonitorSMBG", date_cols=tables["MonitorSMBG"])
    df_MonitorSystem = load_table("MonitorSystem", date_cols=tables["MonitorSystem"])
    df_MonitorTotalBolus = load_table("MonitorTotalBolus", date_cols=tables["MonitorTotalBolus"])

    # Keep only relevant columns
    df_MonitorBasalBolus = df_MonitorBasalBolus[["DeidentID", "LocalDeliveredDtTm", "DeliveredValue"]]
    df_MonitorCGM = df_MonitorCGM[["DeidentID", "LocalDtTm", "CGM"]]
    df_MonitorCorrectionBolus = df_MonitorCorrectionBolus[["DeidentID", "LocalDeliveredDtTm", "BolusSource", "DeliveredValue"]]
    df_MonitorMeal = df_MonitorMeal[["DeidentID", "LocalDtTm", "MealSize", "SMBG"]]
    df_MonitorMealBolus = df_MonitorMealBolus[["DeidentID", "LocalDeliveredDtTm", "DeliveredValue"]]
    df_MonitorSMBG = df_MonitorSMBG[df_MonitorSMBG["IsCalibration"] == 0][["DeidentID", "LocalDtTm", "SMBG", "IsHypo", "DidTreat", "Carbs"]]
    df_MonitorSystem = df_MonitorSystem[["DeidentID", "LocalDtTm", "DiAsState", "IOBValue", "Hypolight", "Hyperlight", "Exercising"]]
    df_MonitorTotalBolus = df_MonitorTotalBolus[["DeidentID", "LocalDeliveredDtTm", "DeliveredValue"]]

    # Write missing values ratio for every column to a file
    with open("preprocessing_log.txt", "w") as f:
        for name, df_ in [("MonitorBasalBolus", df_MonitorBasalBolus), 
                          ("MonitorCGM", df_MonitorCGM), 
                          ("MonitorCorrectionBolus", df_MonitorCorrectionBolus),
                          ("MonitorMeal", df_MonitorMeal), 
                          ("MonitorMealBolus", df_MonitorMealBolus),
                          ("MonitorSMBG", df_MonitorSMBG),
                          ("MonitorSystem", df_MonitorSystem),
                          ("MonitorTotalBolus", df_MonitorTotalBolus)]:
            
            f.write(f"Missing values ratio in {name}:\n")
            # Compute column-wise missingness ratios
            column_missing_ratios = df_.isnull().sum() / len(df_)
            
            # Write each column's ratio to the file
            for col, ratio in column_missing_ratios.items():
                f.write(f"  {col}: {ratio}\n")
            f.write("\n")


    # Unify datetime columns
    for df_bolus in [df_MonitorBasalBolus, df_MonitorCorrectionBolus, df_MonitorMealBolus, df_MonitorTotalBolus]:
        if 'LocalDeliveredDtTm' in df_bolus.columns:
            df_bolus['LocalDtTm'] = df_bolus['LocalDeliveredDtTm']
            df_bolus.drop(columns='LocalDeliveredDtTm', inplace=True)

    # Combine insulin data
    df_Insulin = pd.concat([df_MonitorBasalBolus[['DeidentID', 'LocalDtTm', 'DeliveredValue']],
                            df_MonitorCorrectionBolus[['DeidentID', 'LocalDtTm', 'DeliveredValue']],
                            df_MonitorMealBolus[['DeidentID', 'LocalDtTm', 'DeliveredValue']]])
    df_Insulin = df_Insulin.sort_values(by='LocalDtTm').reset_index(drop=True)
    df_MonitorCGM["LocalDtTm"] = pd.to_datetime(df_MonitorCGM["LocalDtTm"])

    # Remove unrealistic values
    with open("preprocessing_log.txt", "a") as f:
        f.write("Removing unrealistic values...\n")
        df_insulin_shape_pre = df_Insulin.shape[0]
        df_smbg_shape_pre = df_MonitorSMBG.shape[0]
        df_meal_shape_pre = df_MonitorMeal.shape[0]
        df_system_shape_pre = df_MonitorSystem.shape[0]
        df_MonitorCGM_shape_pre = df_MonitorCGM.shape[0]
        
        realistic_insulin_min = 0
        realistic_insulin_max = 1000
        realistic_smbg_min = 0
        realistic_smbg_max = 1000
        realistic_meal_min = 0
        realistic_meal_max = 2000
        realistic_system_min = 0
        realistic_system_max = 100
        
        realistic_cgm_min = 10
        realistic_cgm_max = 401
        
        
        df_Insulin = df_Insulin[(df_Insulin["DeliveredValue"] >= realistic_insulin_min) & (df_Insulin["DeliveredValue"] <= realistic_insulin_max)]
        df_MonitorSMBG = df_MonitorSMBG[(df_MonitorSMBG["Carbs"] >= realistic_smbg_min) & (df_MonitorSMBG["Carbs"] <= realistic_smbg_max)]
        df_MonitorMeal = df_MonitorMeal[(df_MonitorMeal["MealSize"] >= realistic_meal_min) & (df_MonitorMeal["MealSize"] <= realistic_meal_max)]
        df_MonitorSystem = df_MonitorSystem[(df_MonitorSystem["Exercising"] >= realistic_system_min) & (df_MonitorSystem["Exercising"] <= realistic_system_max)]
        df_MonitorCGM = df_MonitorCGM[(df_MonitorCGM["CGM"] >= realistic_cgm_min) & (df_MonitorCGM["CGM"] <= realistic_cgm_max)]
        df_insulin_shape_post = df_Insulin.shape[0]
        df_smbg_shape_post = df_MonitorSMBG.shape[0]
        df_meal_shape_post = df_MonitorMeal.shape[0]
        df_system_shape_post = df_MonitorSystem.shape[0]
        df_MonitorCGM_shape_post = df_MonitorCGM.shape[0]
        
        f.write(f"Insulin shape before: {df_insulin_shape_pre}, after: {df_insulin_shape_post}\n")
        f.write(f"SMBG shape before: {df_smbg_shape_pre}, after: {df_smbg_shape_post}\n")
        f.write(f"Meal shape before: {df_meal_shape_pre}, after: {df_meal_shape_post}\n")
        f.write(f"System shape before: {df_system_shape_pre}, after: {df_system_shape_post}\n")
        f.write(f"CGM shape before: {df_MonitorCGM_shape_pre}, after: {df_MonitorCGM_shape_post}\n")
        f.write("\n")
        
        df_names = ["Insulin", "SMBG", "Meal", "System", "CGM"]
        
        for i, df_ in enumerate([df_Insulin, df_MonitorSMBG, df_MonitorMeal, df_MonitorSystem, df_MonitorCGM]):
            f.write(f"Description of df_{df_names[i]}:\n")
            f.write(f"{df_.describe()}\n\n")
            f.write("\n")
        
    # Generate basic and self-supervised dataframes
    basic_dfs = []
    self_sup_dfs = []
    
    #patients_to_exclude = [1, 9, 10, 12, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30]
    patients = range(1, 31)
    #patients = [p for p in patients if p not in patients_to_exclude]

    for i in patients:
        # Filter by patient
        df_cgm_p = df_MonitorCGM[df_MonitorCGM["DeidentID"] == i].sort_values("LocalDtTm")
        if df_cgm_p.shape[0] == 0:
            with open("preprocessing_log.txt", "a") as f:
                f.write(f"Patient {i} has no CGM data. Skipping...\n")
            continue
        df_meal_p = df_MonitorMeal[df_MonitorMeal["DeidentID"] == i]
        df_smbg_p = df_MonitorSMBG[df_MonitorSMBG["DeidentID"] == i]
        df_system_p = df_MonitorSystem[df_MonitorSystem["DeidentID"] == i]
        df_insulin_p = df_Insulin[df_Insulin["DeidentID"] == i]

        # Floor times to 5min intervals and aggregate
        df_cgm_p["LocalDtTm"] = df_cgm_p["LocalDtTm"].dt.floor("5min")
        df_insulin_p["LocalDtTm"] = df_insulin_p["LocalDtTm"].dt.floor("5min")
        df_smbg_p["LocalDtTm"] = df_smbg_p["LocalDtTm"].dt.floor("5min")
        df_meal_p["LocalDtTm"] = df_meal_p["LocalDtTm"].dt.floor("5min")
        df_system_p["LocalDtTm"] = df_system_p["LocalDtTm"].dt.floor("5min")

        df_insulin_agg = df_insulin_p.groupby("LocalDtTm")["DeliveredValue"].sum()
        df_smbg_agg = df_smbg_p.groupby("LocalDtTm")["Carbs"].sum()
        df_meal_agg = df_meal_p.groupby("LocalDtTm")["MealSize"].sum()
        df_system_agg = df_system_p.groupby("LocalDtTm")["Exercising"].sum()
        df_cgm_agg = df_cgm_p.set_index("LocalDtTm")["CGM"]

        # Create final DF (just times and CGM so far)
        df_final = df_cgm_p[["LocalDtTm", "CGM"]].copy()

        # Check CGM past 288 windows
        df_final_cgm = df_final["LocalDtTm"].progress_apply(lambda x: pd.Series(get_24_hour_bins(df_cgm_agg, x, "CGM")))
        df_final_cgm.columns = [f"cgm_{j}" for j in range(1, 289)]
        df_final["non_imputed_cgm_count"] = df_final_cgm.apply(lambda row: (row != 0).sum(), axis=1)
        df_shape_pre = df_final.shape[0]        
        cgm_min = 280
        df_final = df_final[df_final["non_imputed_cgm_count"] >= cgm_min]
        df_shape_post = df_final.shape[0]
        with open("preprocessing_log.txt", "a") as f:
            f.write(f"Dropped {df_shape_pre - df_shape_post} entries for Patient {i} with less than {cgm_min} non-imputed CGM values.\n")
        df_final = df_final.drop(columns=["non_imputed_cgm_count"])
        # Add Insulin past 288 windows
        df_final_insulin = df_final["LocalDtTm"].progress_apply(lambda x: pd.Series(get_24_hour_bins(df_insulin_agg, x, "insulin")))
        df_final_insulin.columns = [f"insulin_{j}" for j in range(1, 289)]

        # Add Meal past 288 windows
        df_final_meal = df_final["LocalDtTm"].progress_apply(lambda x: pd.Series(get_24_hour_bins(df_meal_agg, x, "Meal")))
        df_final_meal.columns = [f"mealsize_{j}" for j in range(1, 289)]

        # Add SMBG (carbs) past 288 windows
        df_final_smbg = df_final["LocalDtTm"].progress_apply(lambda x: pd.Series(get_24_hour_bins(df_smbg_agg, x, "SMBG")))
        df_final_smbg.columns = [f"carbs_{j}" for j in range(1, 289)]

        # Add System (exercise) past 288 windows
        df_final_system = df_final["LocalDtTm"].progress_apply(lambda x: pd.Series(get_24_hour_bins(df_system_agg, x, "System")))
        df_final_system.columns = [f"exercise_{j}" for j in range(1, 289)]

        df_final = pd.concat([df_final, df_final_insulin, df_final_meal, df_final_smbg, df_final_system], axis=1)

        with open("preprocessing_log.txt", "a") as f:
            f.write(f"Patient {i} final dataframe shape: {df_final.shape}\n")

        df_final = df_final.dropna().sort_values("LocalDtTm").reset_index(drop=True)

        with open("preprocessing_log.txt", "a") as f:
            f.write(f"Patient {i} final dataframe shape after dropping NAs: {df_final.shape}\n")


        
        df_final_self = get_self_sup_df(df_final)
        df_final["DeidentID"] = i
        df_final_self["DeidentID"] = i
        basic_dfs.append(df_final)
        self_sup_dfs.append(df_final_self)
        print(f"Patient {i} processing complete.")

    print("Combining all patient dfs...")
    combine_dfs(basic_dfs, self_sup_dfs)
    print("Final dataframes saved!")


if __name__ == "__main__":
    preprocess_data()
