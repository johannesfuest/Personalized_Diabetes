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
import sys
from tqdm import tqdm

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
    X_train, X_test, Y_train, Y_test = sf.get_train_test_split_all(
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


def load_data_train_model(fixed_hyperparameters, data, CONV_INPUT_LENGTH, write_preds=False):
    """
    Loads the data and trains baseline 1, logging the results to sigopt and writing predictions to files
    :param fixed_hyperparameters: Dict of hyperparameters to use for training
    :param data: X_train, X_test, Y_train, Y_test as pandas dataframes
    :param CONV_INPUT_LENGTH: int describing the length of the input to the convolutional layer
    :param write_preds: Bool describing whether to write predictions to files
    :return: void, but writes predictions to files if write_preds is True
    """
    X_train, X_test, Y_train, Y_test = data
    # create the model
    with tf.device("/GPU:0"):
        model = sf.GlucoseModel(CONV_INPUT_LENGTH, False, fixed_hyperparameters)
    # train the model
    with tf.device("/GPU:0"):
        model.train_model(
            fixed_hyperparameters["num_epochs"],
            X_train,
            X_test,
            Y_train,
            Y_test,
            fixed_hyperparameters["learning_rate"],
            int(fixed_hyperparameters["batch_size"]),
            False,
            fixed_hyperparameters["missingness_modulo"],
            "Sup. DL",
        )
    
        train_gmse, train_mse = model.evaluate_model(X_train, Y_train)
        test_gmse, test_mse = model.evaluate_model(X_test, Y_test)
    train_preds = pd.DataFrame(model.model.predict(X_train))
    train_preds["y"] = Y_train["CGM"].values
    
    train_preds["experiment"] = "baseline_1_train"
    test_preds = pd.DataFrame(model.model.predict(X_test))
    test_preds["y"] = Y_test["CGM"].values
    test_preds["experiment"] = "baseline_1_test"
    print(test_preds.columns)
    # bootstrap 95% CIs for gmse
    n_bootstraps = 1000
    bootstrapped_gmse = np.zeros(n_bootstraps)
    for i in tqdm(range(n_bootstraps)):
        bootstrapped_data = test_preds.sample(frac=1, replace=True)
        bootstrapped_gmse[i] = sf.gMSE(
            bootstrapped_data["y"].values, bootstrapped_data[0].values
        )
        # bootstrap 95% CI of test set performance
    print(f"Bootstrapped 95% CI of test gMSE: {np.percentile(bootstrapped_gmse, [2.5, 97.5])}")
    with open(f"baseline_1_{fixed_hyperparameters['missingness_modulo']}.txt", "w") as f:
        f.write(f"Bootstrapped 95% CI of test gMSE: {np.percentile(bootstrapped_gmse, [2.5, 97.5])}\n")
        f.write(f"Train gMSE: {train_gmse}, Test gMSE: {test_gmse}\n")
    
    if write_preds:
        if not os.path.exists("preds"):
            os.mkdir("preds")
        train_preds.to_csv(
            os.path.join("preds", f"base_1_train_M{fixed_hyperparameters['missingness_modulo']}.csv")
        )
        test_preds.to_csv(
            os.path.join("preds", f"base_1_test_M{fixed_hyperparameters['missingness_modulo']}.csv")
        )
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
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if not name:
        name = ""

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if args.experiment:
        fixed_hyperparameters = {
            "dropout_rate": 0.047392102582145226,
            "learning_rate": 0.0014043829620690904,
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

        missingness_modulos = [10, 20, 50, 100, 200, 400, 800, 1000, 1500, 2000]
        for mm in missingness_modulos:
            data = load_data(0.8, mm)
            fixed_hyperparameters["missingness_modulo"] = mm
            print(f"GPUs available:{tf.config.list_physical_devices('GPU')}")
            
            load_data_train_model(fixed_hyperparameters, data, CONV_INPUT_LENGTH, write_preds=True)
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
        data = load_data(0.8, 100)
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
            budget=20,
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
