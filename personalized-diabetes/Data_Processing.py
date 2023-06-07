import os
import pandas as pd
from tqdm.auto import tqdm
from utils.loading import load_table
import sys
import warnings


pd.set_option("display.max_rows", 4000)
pd.set_option("mode.chained_assignment", None)
sys.path.append("..")
tqdm.pandas()
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def get_24_hour_bins(df, date, df_name):
    """
    This function takes in a dataframe that has a datetime column, as well as some value of note, like delivered insulin.
    It also takes in a date, which is the date of the CGM values in our study. It returns a list of the cumulative
    value of the time series variable over the past 24 hours, binned into 5-minute intervals. If there is no data for a
    given interval, the value is 0. The list is reversed so that the most recent data is first.
    :param df: The dataframe for which the data is to be collected. Must have a 'LocalDtTm' column, as well as a second
    column with the relevant time series variable of the dataframe (e.g. 'CGM', 'SMBG', 'Meal', 'System')
    :param date: The date for which the data over the past 24 hours is to be collected. Must be a datetime object and
    will be the datetime of CGM values in our study.
    :param df_name: Name of the dataframe to identify what the relevant column is called. Must be a string.
    :return: List of the cumulative value of the time series variable over the past 24 hours, binned into 5-minute
    intervals. If there is no data for a given interval, the value is 0. The list is reversed so that the most recent
    data is first.
    """

    # filter the dataframe to keep only the rows within the past 24 hours
    df = df.loc[date - pd.Timedelta(days=1) : date]
    # create a list of 288 values with 0 for any missing intervals
    date_range = pd.date_range(date - pd.Timedelta(days=1), date, freq="5min")
    date_range = date_range.floor("5min")
    # reverse the list
    date_range = date_range[::-1]
    # change date range type
    date_range = date_range.astype(str)
    # make date_range a list of strings
    date_range = date_range.tolist()
    # get the values from the dataframe
    if df_name == "insulin":
        result = [df.DeliveredValue.get(dates, default=0) for dates in date_range]
    elif df_name == "SMBG":
        result = [df.Carbs.get(dates, default=0) for dates in date_range]
    elif df_name == "Meal":
        result = [df.MealSize.get(dates, default=0) for dates in date_range]
    elif df_name == "System":
        result = [df.Exercising.get(dates, default=0) for dates in date_range]
    # return the list of exactly 288 values
    result = result[:288]
    return result


def get_hourly_coverage(df, date_col, patient_col, threshold):
    """
    This function takes in a dataframe and adds a 'covered' column to every hour that indicates whether data is available
    during that hour.
    :param df: The dataframe to be augmented
    :param date_col: The name of the column in the dataframe that contains the datetime information
    :param patient_col: The column indicating which patient the row belongs to
    :param threshold: The number of 5-minute intervals that must be present over the hour for the hour to be considered
    'covered'
    :return: The augmented dataframe with a covered column
    """

    df = df.copy(deep=True).sort_values(date_col).set_index([date_col])
    df.DeidentID = df.DeidentID.astype(int)
    df = (
        df.groupby(patient_col, as_index=True)
        .resample("1H", level=date_col)[[patient_col]]
        .count()
    )
    df.rename(columns={patient_col: "Entries"}, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"level_0": patient_col}, inplace=True)
    df = (
        df.groupby(patient_col, as_index=False)
        .rolling(window=24, on=date_col, closed="right")
        .sum()
    )
    df["covered"] = df.Entries > threshold
    return df


def remove_unrealistic_entries(df):
    """
    This functions takes in a patient dataframe and removes all unrealistic entries from it.
    :param df: The dataframe to be cleaned
    :return: The resulting cleaned dataframe
    """
    print("Dropping rows damaged insulin values")
    df = df[df[[f"insulin {j}" for j in range(1, 289)]].min(axis=1) >= 0]
    df = df[df[[f"insulin {j}" for j in range(1, 289)]].max(axis=1) < 1000]
    print("Dropping rows with damaged carb values")
    df = df[df[[f"carbs {j}" for j in range(1, 289)]].min(axis=1) >= 0]
    df = df[df[[f"carbs {j}" for j in range(1, 289)]].max(axis=1) < 1000]
    print("Dropping rows with damaged exercise values")
    df = df[df[[f"exercise {j}" for j in range(1, 289)]].min(axis=1) >= 0]
    df = df[df[[f"exercise {j}" for j in range(1, 289)]].max(axis=1) < 100]
    print("Dropping rows with damaged mealsize values")
    df = df[df[[f"mealsize {j}" for j in range(1, 289)]].min(axis=1) >= 0]
    df = df[df[[f"mealsize {j}" for j in range(1, 289)]].max(axis=1) < 1000]
    return df


def get_self_sup_df(df):
    """
    A function that takes a dataframe and returns a dataframe with the next row as the targets.
    :param df: the dataframe to be transformed for self-supervised learning
    :return: a dataframe with the next row as target
    """
    # sort df by LocalDtTm
    df = df.sort_values("LocalDtTm")
    # loop over all columns and shift them by one
    for i in range(1, 289):
        df[f"insulin {i} target"] = df[f"insulin {i}"].shift(-1)
        df[f"mealsize {i} target"] = df[f"mealsize {i}"].shift(-1)
        df[f"carbs {i} target"] = df[f"carbs {i}"].shift(-1)
        df[f"exercise {i} target"] = df[f"exercise {i}"].shift(-1)
    # drop na values
    df = df.dropna()
    return df


def combine_dfs(basics, selfs, test=False):
    """
    This function takes in the basic and self supervised dataframes in a list and combines them into one dataframe
    respectively, which is saved to the current directory
    :param basics: A list of the basic dataframes
    :param selfs: A list of the self supervised dataframes
    :param test: A boolean indicating whether run is a test run with downsampled data
    :return:
    """
    for i in range(30):
        df = basics[i]
        df_self = selfs[i]
        if i == 0:
            all_data = df
            all_data_self = df_self
            all_data["DeidentID"] = 1
            all_data_self["DeidentID"] = 1
        else:
            df["DeidentID"] = i
            df_self["DeidentID"] = i
            all_data = pd.concat([all_data, df], axis=0)
            all_data_self = pd.concat([all_data_self, df_self], axis=0)
        print(f"Patient {i + 1} done!")
    if test:
        all_data.to_csv("basic_0_test.csv", index=False)
        all_data_self.to_csv("self_0_test.csv", index=False)
    else:
        all_data.to_csv("basic_0.csv", index=False)
        all_data_self.to_csv("self_0.csv", index=False)


def preprocess_data(test: bool = False):
    """
    This function creates the basic csvs that are used in the analysis. It creates a csv for each patient that contains
    the following columns:
    - DeidentID: The patient ID
    - LocalDtTm: The date of the CGM values in our study
    - CGM: The CGM value at that time
    - insulin 1-288: The cumulative amount of insulin delivered over the past 24 hours, binned into 5-minute intervals
    - carbs 1-288: The cumulative amount of liquid glucose consumed over the past 24 hours, binned into 5-minute intervals
    - mealsize 1-288: The cumulative size of meals consumed over the past 24 hours, binned into 5-minute intervals
    - exercise 1-288: The cumulative amount of time spent exercising over the past 24 hours, binned into 5-minute intervals
    :param test: Bool to run function in dev mode (100 rows per patient only) to see functionality and debug
    :return: None, but saves 2 csvs to the current directory. 'basic_0.csv' and 'self_0.csv', which are the basic and
    self supervised dataframes respectively including data for all patients in the study.

    :return:
    """
    # Dictionary of tables in original study for convenience
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
    # Load relevant tables from study and keep only relevant columns
    print("Reading in files from public study...")
    df_MonitorBasalBolus = load_table(
        filename="MonitorBasalBolus", date_cols=tables["MonitorBasalBolus"]
    )
    df_MonitorCGM = load_table(filename="MonitorCGM", date_cols=tables["MonitorCGM"])
    df_MonitorCorrectionBolus = load_table(
        filename="MonitorCorrectionBolus", date_cols=tables["MonitorCorrectionBolus"]
    )
    df_MonitorMeal = load_table(filename="MonitorMeal", date_cols=tables["MonitorMeal"])
    df_MonitorMealBolus = load_table(
        filename="MonitorMealBolus", date_cols=tables["MonitorMealBolus"]
    )
    df_MonitorSMBG = load_table(filename="MonitorSMBG", date_cols=tables["MonitorSMBG"])
    df_MonitorSystem = load_table(
        filename="MonitorSystem", date_cols=tables["MonitorSystem"]
    )
    df_MonitorTotalBolus = load_table(
        filename="MonitorTotalBolus", date_cols=tables["MonitorTotalBolus"]
    )
    df_MonitorBasalBolus = df_MonitorBasalBolus[
        ["DeidentID", "LocalDeliveredDtTm", "DeliveredValue"]
    ]
    df_MonitorCGM = df_MonitorCGM[["DeidentID", "LocalDtTm", "CGM"]]
    df_MonitorCorrectionBolus = df_MonitorCorrectionBolus[
        ["DeidentID", "LocalDeliveredDtTm", "BolusSource", "DeliveredValue"]
    ]
    df_MonitorMeal = df_MonitorMeal[["DeidentID", "LocalDtTm", "MealSize", "SMBG"]]
    df_MonitorMealBolus = df_MonitorMealBolus[
        ["DeidentID", "LocalDeliveredDtTm", "DeliveredValue"]
    ]
    df_MonitorSMBG = df_MonitorSMBG[df_MonitorSMBG["IsCalibration"] == 0]
    df_MonitorSMBG = df_MonitorSMBG[
        ["DeidentID", "LocalDtTm", "SMBG", "IsHypo", "DidTreat", "Carbs"]
    ]
    df_MonitorSystem = df_MonitorSystem[
        [
            "DeidentID",
            "LocalDtTm",
            "DiAsState",
            "IOBValue",
            "Hypolight",
            "Hyperlight",
            "Exercising",
        ]
    ]
    df_MonitorTotalBolus = df_MonitorTotalBolus[
        ["DeidentID", "LocalDeliveredDtTm", "DeliveredValue"]
    ]
    df_MonitorCorrectionBolus["LocalDtTm"] = df_MonitorCorrectionBolus[
        "LocalDeliveredDtTm"
    ]
    df_MonitorMealBolus["LocalDtTm"] = df_MonitorMealBolus["LocalDeliveredDtTm"]
    df_MonitorTotalBolus["LocalDtTm"] = df_MonitorTotalBolus["LocalDeliveredDtTm"]
    df_MonitorBasalBolus["LocalDtTm"] = df_MonitorBasalBolus["LocalDeliveredDtTm"]
    df_MonitorCorrectionBolus.drop(columns=["LocalDeliveredDtTm"], inplace=True)
    df_MonitorMealBolus.drop(columns=["LocalDeliveredDtTm"], inplace=True)
    df_MonitorTotalBolus.drop(columns=["LocalDeliveredDtTm"], inplace=True)
    df_MonitorBasalBolus.drop(columns=["LocalDeliveredDtTm"], inplace=True)

    # Combine bolus data into one insulin table
    df_Insulin = pd.concat(
        [df_MonitorBasalBolus, df_MonitorCorrectionBolus, df_MonitorMealBolus]
    )
    df_Insulin = df_Insulin.sort_values(by="LocalDtTm", ascending=True)
    df_Insulin = df_Insulin.reset_index(drop=True)
    df_Insulin = df_Insulin[["LocalDtTm", "DeliveredValue", "DeidentID"]]
    df_MonitorCGM["LocalDtTm"] = pd.to_datetime(df_MonitorCGM["LocalDtTm"])

    # Remove rows that include insufficient data coverage for study
    print("Removing rows with insufficient data coverage")
    investigation_tables = {
        "Insuline": (df_Insulin, 0),
        "MonitorCGM": (df_MonitorCGM, 220),
    }
    result_tables = {}
    for table in investigation_tables.keys():
        df = get_hourly_coverage(
            investigation_tables[table][0],
            "LocalDtTm",
            "DeidentID",
            investigation_tables[table][1],
        )
        result_tables[table] = df

    first_df = list(result_tables.keys())[0]
    res_df = result_tables[first_df]
    res_df.rename(columns={"covered": f"covered_{first_df}"}, inplace=True)

    for table in list(result_tables.keys())[1:]:
        res_df = res_df.merge(
            result_tables[table],
            on=["DeidentID", "LocalDtTm"],
            how="outer",
            suffixes=("", f"_{table}"),
        )
        res_df.rename(columns={"covered": f"covered_{table}"}, inplace=True)
    covered_cols = [x for x in res_df.columns if "covered" in x]
    res_df["all_covered"] = res_df[covered_cols].fillna(False).all(axis=1)

    # Begin individual patientwise dataframe creation
    basic_dfs = []
    self_sup_dfs = []
    for i in range(1, 31):
        print("Starting df creation for patient: ", i, "...")
        DeidentId = str(i)
        # filter by patient
        df_MonitorCGM_filtered = df_MonitorCGM[df_MonitorCGM["DeidentID"] == DeidentId]
        df_MonitorMeal_filtered = df_MonitorMeal[
            df_MonitorMeal["DeidentID"] == DeidentId
        ]
        df_MonitorSMBG_filtered = df_MonitorSMBG[
            df_MonitorSMBG["DeidentID"] == DeidentId
        ]
        df_MonitorSystem_filtered = df_MonitorSystem[
            df_MonitorSystem["DeidentID"] == DeidentId
        ]
        res_df_filtered = res_df[res_df["DeidentID"] == int(DeidentId)]
        df_Insulin_filtered = df_Insulin[df_Insulin["DeidentID"] == DeidentId]

        # filter df_MonitorCGM to only contain rows whose LocalDtTm is within 1 hour of any row in res_df
        res_df_filtered = res_df_filtered[res_df_filtered["all_covered"] == True]
        df_MonitorCGM_filtered = df_MonitorCGM_filtered.sort_values(
            by="LocalDtTm", ascending=True
        )
        res_df_filtered = res_df_filtered.sort_values(by="LocalDtTm", ascending=True)
        df_MonitorCGM_filtered["LocalDtTm"] = pd.to_datetime(
            df_MonitorCGM_filtered["LocalDtTm"]
        )
        res_df_filtered["LocalDtTm"] = pd.to_datetime(res_df_filtered["LocalDtTm"])
        df_MonitorCGM_filtered["rounded"] = df_MonitorCGM_filtered[
            "LocalDtTm"
        ].dt.round("1H")
        res_df_filtered["rounded"] = res_df_filtered["LocalDtTm"].dt.round("1H")
        df_MonitorCGM_filtered = df_MonitorCGM_filtered[
            df_MonitorCGM_filtered["rounded"].isin(res_df_filtered["rounded"])
        ]
        df_MonitorCGM_filtered = df_MonitorCGM_filtered.drop(columns=["rounded"])

        df_Insulin_filtered["LocalDtTm"] = df_Insulin_filtered["LocalDtTm"].dt.floor(
            "5min"
        )
        df_Insulin_agg = (
            df_Insulin_filtered.groupby("LocalDtTm")["DeliveredValue"]
            .sum()
            .astype(float)
        )
        df_MonitorSMBG_filtered["LocalDtTm"] = df_MonitorSMBG_filtered[
            "LocalDtTm"
        ].dt.floor("5min")
        df_MonitorSMBG_agg = (
            df_MonitorSMBG_filtered.groupby("LocalDtTm")["Carbs"].sum().astype(float)
        )
        df_MonitorMeal_filtered["LocalDtTm"] = df_MonitorMeal_filtered[
            "LocalDtTm"
        ].dt.floor("5min")
        df_MonitorMeal_agg = (
            df_MonitorMeal_filtered.groupby("LocalDtTm")["MealSize"].sum().astype(float)
        )
        df_MonitorSystem_filtered["LocalDtTm"] = df_MonitorSystem_filtered[
            "LocalDtTm"
        ].dt.floor("5min")
        df_MonitorSystem_agg = (
            df_MonitorSystem_filtered.groupby("LocalDtTm")["Exercising"]
            .sum()
            .astype(float)
        )
        # make Exercising binary
        df_MonitorSystem_agg = df_MonitorSystem_agg.apply(lambda x: 1 if x > 0 else 0)
        df_Insulin_agg = df_Insulin_agg.to_frame()
        df_MonitorSMBG_agg = df_MonitorSMBG_agg.to_frame()
        df_MonitorMeal_agg = df_MonitorMeal_agg.to_frame()
        df_MonitorSystem_agg = df_MonitorSystem_agg.to_frame()

        df_final = pd.DataFrame()
        df_final["LocalDtTm"] = df_MonitorCGM_filtered["LocalDtTm"]
        df_final["CGM"] = df_MonitorCGM_filtered["CGM"]
        df_final["LocalDtTm"] = pd.to_datetime(df_final["LocalDtTm"])
        if test:
            # reduce df_final to 100 rows for testing
            df_final = df_final.iloc[:100, :]

        print(
            "Starting insulin, mealsize, carbs and exercise calculations for patient: ",
            i,
            "...",
        )
        new_df_insuline = df_final["LocalDtTm"].progress_apply(
            lambda x: pd.Series(get_24_hour_bins(df_Insulin_agg, x, "insulin"))
        )
        new_df_insuline.columns = [f"insulin {i}" for i in range(1, 289)]

        new_df_meal = df_final["LocalDtTm"].progress_apply(
            lambda x: pd.Series(get_24_hour_bins(df_MonitorMeal_agg, x, "Meal"))
        )
        new_df_meal.columns = [f"mealsize {i}" for i in range(1, 289)]
        new_df_smbg = df_final["LocalDtTm"].progress_apply(
            lambda x: pd.Series(get_24_hour_bins(df_MonitorSMBG_agg, x, "SMBG"))
        )
        new_df_smbg.columns = [f"carbs {i}" for i in range(1, 289)]
        new_df_system = df_final["LocalDtTm"].progress_apply(
            lambda x: pd.Series(get_24_hour_bins(df_MonitorSystem_agg, x, "System"))
        )
        new_df_system.columns = [f"exercise {i}" for i in range(1, 289)]
        print(
            "Finished insulin, mealsize, carbs and exercise calculations for patient: ",
            i,
            ". Creating final dataframe...",
        )
        df_final = pd.concat(
            [df_final, new_df_insuline, new_df_meal, new_df_smbg, new_df_system], axis=1
        )
        df_final = df_final.dropna()
        df_final = df_final.reset_index(drop=True)
        df_final["LocalDtTm"] = pd.to_datetime(df_final["LocalDtTm"])
        df_final = df_final.sort_values(by="LocalDtTm")
        print(f"Final df created for patient {DeidentId}")
        df_final = remove_unrealistic_entries(df_final)
        print(f"Final dataframe cleaned of unrealistic entries for patient {DeidentId}")
        df_final_self = get_self_sup_df(df_final)
        df_final["DeidentID"] = DeidentId
        df_final_self["DeidentID"] = DeidentId
        basic_dfs.append(df_final)
        self_sup_dfs.append(df_final_self)
        print(
            f"Self-supervised and basic dfs successfully constructed for patient {DeidentId}"
        )
    print("Combining all dfs...")
    combine_dfs(basic_dfs, self_sup_dfs, test)
    print("Final dataframes saved!")
    return


if __name__ == "__main__":
    preprocess_data()
