# Python file for running baseline 1 (no individualization, no self-supervised data) using sigopt
import argparse
import git
import numpy as np
import os
import pandas as pd
import random
import sigopt_functions as sf
import sigopt
import tensorflow as tf

SEED = 0

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


# Set seeds for reproducibility
set_global_determinism(seed=SEED)

os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"
DATASET = "basic_0.csv"


def load_data(split: float, missingness_modulo: int):
    """
    Loads the data for baseline 1.
    :param split: Float describing how much of the data to use for training
    :param missingness_modulo: Int n describing how much of the data to delete (keep only every nth row)
    :return: X_train, X_test, Y_train, Y_test as pandas dataframes
    """
    df_basic = pd.read_csv(DATASET)
    patients_to_exclude = [1, 9, 10, 12, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30]
    df_basic = df_basic[~df_basic.DeidentID.isin(patients_to_exclude)]
    print("data read")
    X_train, X_test, Y_train, Y_test = sf.get_train_test_split_search(
        df_basic, split, False
    )
    X_train, Y_train = sf.apply_data_missingness(
        x_train=X_train, y_train=Y_train, missingness_modulo=missingness_modulo
    )
    X_train.drop(columns=["DeidentID"], inplace=True)
    X_test.drop(columns=["DeidentID"], inplace=True)
    Y_train.drop(columns=["DeidentID"], inplace=True)
    Y_test.drop(columns=["DeidentID"], inplace=True)
    return X_train, X_test, Y_train, Y_test


def load_data_train_model(run, data, CONV_INPUT_LENGTH, write_preds=False):
    """
    Loads the data and trains baseline 1, logging the results to sigopt and writing predictions to files
    :param run: sigopt run objects with run-specific parameters
    :param data: X_train, X_test, Y_train, Y_test as pandas dataframes
    :param CONV_INPUT_LENGTH: int describing the length of the input to the convolutional layer
    :param write_preds: Bool describing whether to write predictions to files
    :return: void, but writes predictions to files if write_preds is True
    """
    run.log_dataset(name=DATASET)
    X_train, X_test, Y_train, Y_test = data
    # create the model
    with tf.device("/GPU:0"):
        model = sf.GlucoseModel(CONV_INPUT_LENGTH, False, run)
    run.log_model("Baseline 1")
    # train the model
    with tf.device("/GPU:0"):
        model.train_model(
            run.params.num_epochs,
            X_train,
            X_test,
            Y_train,
            Y_test,
            run.params.learning_rate,
            int(run.params.batch_size),
            False,
        )
    run.log_metadata("sgd optimizer", "adam")
    train_gmse, train_mse = model.evaluate_model(X_train, Y_train)
    test_gmse, test_mse = model.evaluate_model(X_test, Y_test)

    print(f"len(x_train){len(X_train)})")
    print(f"len(x_test){len(X_test)})")
    print(f"train_mse{train_mse})")
    print(f"train_gme{train_gmse})")
    print(f"test_mse{test_mse})")
    print(f"test_gme{test_gmse})")

    print("Y-TRAIN:")
    print(Y_train.describe())
    print("Y-HAT-TRAIN:")
    train_preds = pd.DataFrame(model.model.predict(X_train))
    train_preds["y"] = Y_train["CGM"].values
    print(train_preds.describe())

    train_preds["run"] = run.id
    train_preds["experiment"] = run.experiment
    print("Y-TEST:")
    print(Y_test.describe())
    print("Y-HAT-TEST:")
    test_preds = pd.DataFrame(model.model.predict(X_test))
    test_preds["y"] = Y_test["CGM"].values
    test_preds["run"] = run.id
    test_preds["experiment"] = run.experiment
    print(test_preds.describe())

    if write_preds:
        if not os.path.exists("preds"):
            os.mkdir("preds")
        train_preds.to_csv(
            os.path.join("preds", f"base_1_train_M{run.params.missingness_modulo}.csv")
        )
        test_preds.to_csv(
            os.path.join("preds", f"base_1_test_M{run.params.missingness_modulo}.csv")
        )

    # log performance metrics
    run.log_metric("train gMSE", train_gmse)
    run.log_metric("train MSE", train_mse)
    run.log_metric("test gMSE", test_gmse)
    run.log_metric("test MSE", test_mse)
    tf.keras.backend.clear_session()
    return

if __name__ == "__main__":
    # Either runs experiment or grid search for final model (for experiment use --experiment)
    CONV_INPUT_LENGTH = 288
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Specify an experiment name")
    parser.add_argument(
        "--experiment", action="store_true", help="Enable experiment mode"
    )
    args = parser.parse_args()
    name = args.name
    if not name:
        name = ""

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if args.experiment:
        fixed_hyperparameters = {
            "dropout_rate": 0.0579,
            "learning_rate": 0.001362939,
            "num_epochs": 10,
            "batch_size": 32,
            "filter_1": 4,
            "kernel_1": 6,
            "stride_1": 2,
            "pool_size_1": 2,
            "pool_stride_1": 2,
            "filter_2": 7,
            "kernel_2": 5,
            "stride_2": 2,
            "pool_size_2": 6,
            "pool_stride_2": 5,
        }
        experiment = sigopt.create_experiment(
            name=f"Baseline_1_EXPERIMENT_{name}",
            type="grid",
            parameters=[
                dict(
                    name="missingness_modulo",
                    type="int",
                    grid=[10, 20, 50, 100, 200, 400, 800, 1000, 1500, 2000],
                )
            ],
            metrics=[dict(name="test gMSE", strategy="optimize", objective="minimize")],
            parallel_bandwidth=1,
            # budget=11,
        )

        for run in experiment.loop():
            with run:
                data = load_data(0.8, run.params.missingness_modulo)
                for parameter, value in fixed_hyperparameters.items():
                    run.params[parameter] = value
                    run.log_metadata(parameter, value)
                run.log_metadata("commit", sha)
                run.log_metadata(
                    "GPUs available", tf.config.list_physical_devices("GPU")
                )
                load_data_train_model(run, data, CONV_INPUT_LENGTH, write_preds=True)
    else:
        fixed_hyperparameters = {
            "batch_size": 32,
            "filter_1": 3,
            "kernel_1": 6,
            "stride_1": 2,
            "pool_size_1": 3,
            "pool_stride_1": 2,
            "filter_2": 7,
            "kernel_2": 6,
            "stride_2": 2,
            "pool_size_2": 6,
            "pool_stride_2": 4,
        }
        data = load_data(0.8, 1)
        experiment = sigopt.create_experiment(
            name=f"Baseline_1_{name}",
            type="offline",
            parameters=[
                dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.2)),
                dict(name="learning_rate", type="double", bounds=dict(min=0.0001, max=0.002)),
                dict(name="num_epochs", type="int", bounds=dict(min=5, max=15)),
            ],
            metrics=[dict(name="test gMSE", strategy="optimize", objective="minimize")],
            parallel_bandwidth=1,
            budget=100,
        )
        for run in experiment.loop():
            with run:
                run.log_metadata("commit", sha)
                run.log_metadata(
                    "GPUs available", tf.config.list_physical_devices("GPU")
                )
                for parameter, value in fixed_hyperparameters.items():
                    run.params[parameter] = value
                    run.log_metadata(parameter, value)
                load_data_train_model(run, data, CONV_INPUT_LENGTH)
