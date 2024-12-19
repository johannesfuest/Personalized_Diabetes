import pandas as pd


def get_train_test_split_single_patient(df, TRAIN_TEST_SPLIT: float, self_sup: bool):
    """
    A function that takes a dataframe and returns a temporal train and test split using the given argument to determine
    split.
    :param df: the dataframe to be split
    :param TRAIN_TEST_SPLIT: the split to be used as a number between 0 and 1
    :param self_sup: Boolean indicating whether dataframe is self-supervised or not
    :return: X_train, X_test, Y_train, Y_test according to the given split
    """
    df = df.sort_values("LocalDtTm")
    # split into train and test
    train_length = int((TRAIN_TEST_SPLIT**2) * df.shape[0])
    val_length = int(TRAIN_TEST_SPLIT * df.shape[0])
    train = df.iloc[:train_length, :]
    val = df.iloc[train_length:val_length, :]
    test = df.iloc[val_length:, :]
    # Sanity check: train and test should add up to the original dataframe
    assert (
        test.shape[0] + train.shape[0] + val.shape[0] == df.shape[0]
    ), "Train-Test shapes don not add up."
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
        Y_train = train.drop(columns=["LocalDtTm", "CGM"])
        Y_val = val.drop(columns=["LocalDtTm", "CGM"])
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
        Y_test = test.drop(columns=["LocalDtTm", "CGM"])
        for i in range(1, 289):
            Y_train = Y_train.drop(
                columns=[f"insulin_{i}", f"mealsize_{i}", f"carbs_{i}", f"exercise_{i}"]
            )
            Y_test = Y_test.drop(
                columns=[f"insulin_{i}", f"mealsize_{i}", f"carbs_{i}", f"exercise_{i}"]
            )
            Y_val = Y_val.drop(
                columns=[f"insulin_{i}", f"mealsize_{i}", f"carbs_{i}", f"exercise_{i}"]
            )
    else:
        # Drop the columns that are not needed for supervised learning in train set
        X_train = train.drop(columns=["LocalDtTm", "CGM"])
        X_val = val.drop(columns=["LocalDtTm", "CGM"])
        X_test = test.drop(columns=["LocalDtTm", "CGM"])
        # Drop the columns that are not needed for supervised learning in test set
        Y_train = train[["CGM", "DeidentID"]]
        Y_val = val[["CGM", "DeidentID"]]
        Y_test = test[["CGM", "DeidentID"]]
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def get_train_test_split_across_patients(df, TRAIN_TEST_SPLIT: float, self_sup: bool):
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
            df[df["DeidentID"] == i], TRAIN_TEST_SPLIT, self_sup
        )
        X_train = pd.concat([X_train, X_train_temp])
        Y_train = pd.concat([Y_train, Y_train_temp])
        X_val = pd.concat([X_val, X_val_temp])
        Y_val = pd.concat([Y_val, Y_val_temp])
        X_test = pd.concat([X_test, X_test_temp])
        Y_test = pd.concat([Y_test, Y_test_temp])
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def apply_data_missingness(x_train, y_train, missingness_modulo: int, offset: int = 0):
    """
    A function that applies missingness to the data according to the given modulo. Data missingness is achieved by
    removing every nth row from the data to arrive at an evenly spaced missingness pattern. This is done to reflect
    the likely use case for our model (i.e. one CGM measurement every week instead of every 5 minutes).
    :param x_train: The input data
    :param y_train: The target data
    :param missingness_modulo: How many rows to skip
    :param offset: How many rows to skip from the beginning
    :return: x_train, y_train with missingness applied
    """
    # Sanity check: x_train and y_train should have the same number of rows
    assert (
        x_train.shape[0] == y_train.shape[0]
    ), "x_train and y_train should have the same number of rows before missingness is applied."
    assert (
        offset < x_train.shape[0]
    ), "Offset should be less than the number of rows in the dataset."
    assert (
        offset < missingness_modulo
    ), "Offset should be less than the missingness modulo."
    # Apply offset
    x_train = x_train[offset:]
    y_train = y_train[offset:]
    x_train = x_train[::missingness_modulo]
    y_train = y_train[::missingness_modulo]
    assert (
        x_train.shape[0] == y_train.shape[0]
    ), "x_train and y_train should have the same number of rows after missingness is applied."
    return x_train, y_train