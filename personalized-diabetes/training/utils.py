import pandas as pd
import numpy as np
import torch
import random
import os


def get_train_test_split_single_patient(
    df,
    TRAIN_TEST_SPLIT: float,
    self_sup: bool,
    missingness_modulo: int = 1,
    offset: int = 0
):
    """
    A function that takes a dataframe and returns a temporal train, val, and test split 
    using the given argument to determine the split.

    :param df: the dataframe to be split
    :param TRAIN_TEST_SPLIT: the split to be used as a number between 0 and 1
    :param self_sup: Boolean indicating whether dataframe is self-supervised or not
    :param missingness_modulo: An integer n indicating that only every n-th row 
           should be kept for the train and val sets. Defaults to 1 (i.e. keep all).
    :param offset: An integer indicating how many rows to skip before applying the 
           every-nth-row filter. Defaults to 0.

    :return: X_train, X_val, X_test, Y_train, Y_val, Y_test 
             according to the given split
    """
    # 1) Sort the dataframe by time
    df = df.sort_values("LocalDtTm")

    # 2) Split into train, val, test
    train_length = int((TRAIN_TEST_SPLIT**2) * df.shape[0])
    val_length = int(TRAIN_TEST_SPLIT * df.shape[0])

    train = df.iloc[:train_length, :]
    val = df.iloc[train_length:val_length, :]
    test = df.iloc[val_length:, :]

    # Sanity check
    assert (
        test.shape[0] + train.shape[0] + val.shape[0] == df.shape[0]
    ), "Train-Val-Test shapes do not add up."

    # 3) Drop columns depending on self_sup
    if self_sup:
        # Drop the columns that are not needed for self-supervised learning
        X_train = train.drop(
            columns=[
                "LocalDtTm",
                "CGM",
                "future_insulin",
                "future_meal",
                "future_carbs",
                "future_exercise",
            ]
        )
        X_val = val.drop(
            columns=[
                "LocalDtTm",
                "CGM",
                "future_insulin",
                "future_meal",
                "future_carbs",
                "future_exercise",
            ]
        )
        X_test = test.drop(
            columns=[
                "LocalDtTm",
                "CGM",
                "future_insulin",
                "future_meal",
                "future_carbs",
                "future_exercise",
            ]
        )

        Y_train = train.drop(columns=["LocalDtTm", "CGM"])
        Y_val = val.drop(columns=["LocalDtTm", "CGM"])
        Y_test = test.drop(columns=["LocalDtTm", "CGM"])

        # For self-supervised, drop the columns for insulin_i, mealsize_i, carbs_i, and exercise_i
        for i in range(1, 289):
            for c in [f"insulin_{i}", f"mealsize_{i}", f"carbs_{i}", f"exercise_{i}"]:
                if c in Y_train.columns:
                    Y_train = Y_train.drop(columns=[c])
                if c in Y_val.columns:
                    Y_val = Y_val.drop(columns=[c])
                if c in Y_test.columns:
                    Y_test = Y_test.drop(columns=[c])

    else:
        # Drop the columns that are not needed for supervised learning in train/val
        X_train = train.drop(columns=["LocalDtTm", "CGM"])
        X_val   = val.drop(columns=["LocalDtTm", "CGM"])
        X_test  = test.drop(columns=["LocalDtTm", "CGM"])

        # Y targets are just the CGM + DeidentID
        Y_train = train[["CGM", "DeidentID"]]
        Y_val   = val[["CGM", "DeidentID"]]
        Y_test  = test[["CGM", "DeidentID"]]

    # 4) Apply missingness_modulo + offset to train and val only
    #    Keep only every n-th row starting at 'offset'
    if missingness_modulo > 1 or offset > 0:
        X_train = X_train.iloc[offset::missingness_modulo]
        Y_train = Y_train.iloc[offset::missingness_modulo]
        X_val   = X_val.iloc[offset::missingness_modulo]
        Y_val   = Y_val.iloc[offset::missingness_modulo]
        # Test set remains untouched

    # 5) Return the subsets
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def get_train_test_split_across_patients(df, TRAIN_TEST_SPLIT: float, self_sup: bool, missingness_modulo: int = 1, offset: int = 0):
    """
    A function that takes a dataframe and returns a temporal train and test split using the given argument to determine
    :param df: The dataframe to be split
    :param TRAIN_TEST_SPLIT: The split to be used as a number between 0 and 1
    :param self_sup: Whether the dataframe is self-supervised or not
    :return: X_train, X_test, Y_train, Y_test according to the given split
    """
    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_val = pd.DataFrame()
    Y_val = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()
    for i in range(1, 31):
        X_train_temp, X_val_temp, X_test_temp, Y_train_temp, Y_val_temp, Y_test_temp = get_train_test_split_single_patient(
            df[df["DeidentID"] == i], TRAIN_TEST_SPLIT, self_sup, missingness_modulo, offset
        )
        X_train = pd.concat([X_train, X_train_temp])
        Y_train = pd.concat([Y_train, Y_train_temp])
        X_val = pd.concat([X_val, X_val_temp])
        Y_val = pd.concat([Y_val, Y_val_temp])
        X_test = pd.concat([X_test, X_test_temp])
        Y_test = pd.concat([Y_test, Y_test_temp])
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def set_global_seed(seed: int):
    """
    Sets a single seed for Python, NumPy, and PyTorch to ensure reproducibility
    across multiple runs.
    """
    # 1) Python's built-in random
    random.seed(seed)
    
    # 2) NumPy
    np.random.seed(seed)
    
    # 3) PyTorch CPU
    torch.manual_seed(seed)
    
    # 4) PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5) Set PyTorch to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optionally, set env variable for even more reproducibility:
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Global seed set to {seed}. Deterministic mode enabled for PyTorch.")
