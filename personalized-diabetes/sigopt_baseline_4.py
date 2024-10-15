# Python file for running baseline 4 (individualization, self-supervised data) using sigopt
import argparse
import git
import numpy as np
import os
import pandas as pd
import random
import sigopt_functions as sf
import sigopt
import tensorflow as tf
import gc
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
DATASET_SELF = "self_0.csv"


def load_data(split: float, missingness_modulo: int, search: bool):
    """
    Load data for baseline 4
    :param split: Float describing how much of the data to use for training
    :param missingness_modulo: Int n describing how much of the data to delete (keep only every nth row)
    :return: X_train, X_test, Y_train, Y_test, X_train_self, X_test_self, Y_train_self, Y_test_self as pandas dataframes
    """
    df_basic = pd.read_csv(DATASET)
    patients_to_exclude = [1, 9, 10, 12, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30]
    df_basic = df_basic[~df_basic.DeidentID.isin(patients_to_exclude)]
    print("Basic data read")
    df_self = pd.read_csv(DATASET_SELF)
    df_self = df_self[~df_self.DeidentID.isin(patients_to_exclude)]
    print("Self data read")
    if search:
        df_self = df_self.sample(frac=0.1)
    # delete a fraction of the df rows according to data_missingness
    X_train, X_test, Y_train, Y_test = sf.get_train_test_split_all(
        df_basic, split, False
    )
    X_train, Y_train = sf.apply_data_missingness(
        x_train=X_train, y_train=Y_train, missingness_modulo=missingness_modulo
    )
    (
        X_train_self,
        X_test_self,
        Y_train_self,
        Y_test_self,
    ) = sf.get_train_test_split_all(df_self, split, True)

    return (
        X_train,
        X_test,
        Y_train,
        Y_test,
        X_train_self,
        X_test_self,
        Y_train_self,
        Y_test_self,
    )


def load_data_train_model(fixed_hyperparameters, data, CONV_INPUT_LENGTH, write_preds=False):
    """
    Loads the data and trains the baseline 4, logging the results to sigopt and writing predictions to files
    :param run: sigopt run objects with run-specific parameters
    :param data: X_train, X_test, Y_train, Y_test, X_train_self, X_test_self, Y_train_self, Y_test_self as pandas dataframes
    :param CONV_INPUT_LENGTH: int describing the length of the input to the convolutional layer
    :param write_preds: Bool describing whether to write predictions to files
    :return: void, but writes predictions to files if write_preds is True
    """
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        X_train_self,
        X_test_self,
        Y_train_self,
        Y_test_self,
    ) = data
    weights_train = []
    weights_test = []
    train_mses = []
    train_gmses = []
    test_mses = []
    test_gmses = []
    bootstrap_lower = []
    bootstrap_upper = []
    patients_to_exclude = [1, 9, 10, 12, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30]
    patients_to_include = [i for i in range(1, 31) if i not in patients_to_exclude]
    for i in patients_to_include:
        # create the model
        with tf.device("/GPU:0"):
            model = sf.GlucoseModel(CONV_INPUT_LENGTH, True, fixed_hyperparameters)
        x_train = X_train_self[X_train_self["DeidentID"] == i]
        x_test = X_test_self[X_test_self["DeidentID"] == i]
        y_train = Y_train_self[Y_train_self["DeidentID"] == i]
        y_test = Y_test_self[Y_test_self["DeidentID"] == i]
        x_train = x_train.drop(columns=["DeidentID"])
        x_test = x_test.drop(columns=["DeidentID"])
        y_train = y_train.drop(columns=["DeidentID"])
        y_test = y_test.drop(columns=["DeidentID"])
        with tf.device("/GPU:0"):
            model.train_model(
                fixed_hyperparameters["num_epochs_1"],
                x_train,
                x_test,
                y_train,
                y_test,
                fixed_hyperparameters["learning_rate_1"],
                int(fixed_hyperparameters["batch_size"]),
                True,
                fixed_hyperparameters["missingness_modulo"],
                "Fully Indiv. self-sup. DL",
            )

            # individualization
            model.activate_finetune_mode()
        x_train = X_train[X_train["DeidentID"] == i]
        x_test = X_test[X_test["DeidentID"] == i]
        y_train = Y_train[Y_train["DeidentID"] == i]
        y_test = Y_test[Y_test["DeidentID"] == i]
        x_train = x_train.drop(columns=["DeidentID"])
        x_test = x_test.drop(columns=["DeidentID"])
        y_train = y_train.drop(columns=["DeidentID"])
        y_test = y_test.drop(columns=["DeidentID"])
        with tf.device("/GPU:0"):
            model.train_model(
                fixed_hyperparameters["num_epochs_2"],
                x_train,
                x_test,
                y_train,
                y_test,
                fixed_hyperparameters["learning_rate_2"],
                int(fixed_hyperparameters["batch_size"]),
                False,
                fixed_hyperparameters["missingness_modulo"],
                "Fully Indiv. self-sup. DL",
            )
            # evaluate the model
            # evaluate the model
            train_gmse, train_mse = model.evaluate_model(x_train, y_train)
            test_gmse, test_mse = model.evaluate_model(x_test, y_test)
            
            train_preds = pd.DataFrame(model.model.predict(x_train))
            train_preds["y"] = y_train["CGM"].values
            test_preds = pd.DataFrame(model.model.predict(x_test))
            test_preds["y"] = y_test["CGM"].values
            # bootstrap 95% CI of test set performance
            n_bootstraps = 1000
            bootstrapped_gmse = np.zeros(n_bootstraps)
            for j in tqdm(range(n_bootstraps)):
                bootstrapped_data = test_preds.sample(frac=1, replace=True)
                bootstrapped_gmse[j] = sf.gMSE(
                    bootstrapped_data["y"].values, bootstrapped_data[0].values
                )
            # bootstrapped_gmse = np.clip(bootstrapped_gmse, a_max=10000000, a_min=0)
            print(f"Bootstrapped 95% CI of test gMSE: {np.percentile(bootstrapped_gmse, [2.5, 97.5])}")
            bootstrap_lower.append(np.percentile(bootstrapped_gmse, 2.5))
            bootstrap_upper.append(np.percentile(bootstrapped_gmse, 97.5))

        if write_preds:
            if not os.path.exists("preds"):
                os.mkdir("preds")
            train_preds.to_csv(
                os.path.join(
                    "preds", f"base_4_train_M{fixed_hyperparameters['missingness_modulo']}_D{i}.csv"
                )
            )
            test_preds.to_csv(
                os.path.join(
                    "preds", f"base_4_test_M{fixed_hyperparameters['missingness_modulo']}_D{i}.csv"
                )
            )
        # log the model weights
        weights_train.append(len(x_train))
        weights_test.append(len(x_test))
        train_mses.append(train_mse)
        train_gmses.append(train_gmse)
        test_mses.append(test_mse)
        test_gmses.append(test_gmse)

    # train_gmses = np.clip(train_gmses, a_max=10000000, a_min=0)
    # train_mses = np.clip(train_mses, a_max=10000000, a_min=0)
    # test_gmses = np.clip(test_gmses, a_max=10000000, a_min=0)
    # test_mses = np.clip(test_mses, a_max=10000000, a_min=0)


    train_mse = 0
    train_gmse = 0
    test_mse = 0
    test_gmse = 0
    # bootstrap_lower = np.clip(bootstrap_lower, a_max=10000000, a_min=0)
    # bootstrap_upper = np.clip(bootstrap_upper, a_max=10000000, a_min=0)
    bootstrap_lower = np.mean(bootstrap_lower)
    bootstrap_upper = np.mean(bootstrap_upper)
    with open(f"baseline_4_{fixed_hyperparameters['missingness_modulo']}.txt", "w") as f:
        f.write(f"Bootstrapped 95% CI of test gMSE: [{bootstrap_lower}, {bootstrap_upper}]\n")

    for k in range(14):
        train_mse += weights_train[k] * train_mses[k]
        train_gmse += weights_train[k] * train_gmses[k]
        test_mse += weights_test[k] * test_mses[k]
        test_gmse += weights_test[k] * test_gmses[k]
    train_mse /= sum(weights_train)
    train_gmse /= sum(weights_train)
    test_mse /= sum(weights_test)
    test_gmse /= sum(weights_test)
    
    with open(f"baseline_4_{fixed_hyperparameters['missingness_modulo']}.txt", "a") as f:
        f.write(f"Train gMSE: {train_gmse}, Test gMSE: {test_gmse}\n")
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
            "dropout_rate": 0.1264856752292274,
            "learning_rate_1": 0.0008236880714098919,
            "learning_rate_2": 0.002,
            "num_epochs_1": 15,
            "num_epochs_2": 13,
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

        missing_modulos = [10, 20, 50, 100, 200, 400, 800, 1000, 1500, 2000]
        for mm in missing_modulos:
            fixed_hyperparameters["missingness_modulo"] = mm
            data = load_data(0.8, fixed_hyperparameters["missingness_modulo"], False)
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
        data = load_data(0.8, 100, True)
        experiment = sigopt.create_experiment(
            name=f"Baseline_4_{name}",
            type="offline",
            parameters=[
                dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.2)),
                dict(name="learning_rate_1", type="double", bounds=dict(min=0.0001, max=0.002)),
                dict(name="learning_rate_2", type="double", bounds=dict(min=0.0001, max=0.002)),
                dict(name="num_epochs_1", type="int", bounds=dict(min=5, max=15)),
                dict(name="num_epochs_2", type="int", bounds=dict(min=5, max=15)),
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
