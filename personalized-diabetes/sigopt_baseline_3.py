# Python file for running baseline 3 (individualization, no self-supervised data) using sigopt
import pandas as pd
import sigopt_functions as sf
import sigopt
import git
import os
import tensorflow as tf
import argparse


os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"
DATASET = "basic_0.csv"


def load_data(split: float, data_missingness: float):
    df_basic = pd.read_csv(DATASET)
    print("data read")
    # delete a fraction of the df rows according to data_missingness
    df_basic = sf.apply_data_missingness(df_basic, data_missingness)

    X_train, X_test, Y_train, Y_test = sf.get_train_test_split_search(
        df_basic, split, False
    )
    return X_train, X_test, Y_train, Y_test


def load_data_train_model(run, data, CONV_INPUT_LENGTH):
    run.log_dataset(name=DATASET)
    X_train, X_test, Y_train, Y_test = data
    weights_train = []
    weights_test = []
    train_mses = []
    train_gmses = []
    test_mses = []
    test_gmses = []
    run.log_model("Baseline 3")
    run.log_metadata("sgd optimizer", "adam")
    for i in range(1, 31):
        # create the model
        with tf.device("/GPU:0"):
            model = sf.GlucoseModel(CONV_INPUT_LENGTH, False, run)
        x_train = X_train[X_train["DeidentID"] == i]
        x_test = X_test[X_test["DeidentID"] == i]
        y_train = Y_train[Y_train["DeidentID"] == i]
        y_test = Y_test[Y_test["DeidentID"] == i]
        # drop the deidentID column
        x_train = x_train.drop(columns=["DeidentID"])
        x_test = x_test.drop(columns=["DeidentID"])
        y_train = y_train.drop(columns=["DeidentID"])
        y_test = y_test.drop(columns=["DeidentID"])
        # train the model
        with tf.device("/GPU:0"):
            model.train_model(
                run.params.num_epochs,
                x_train,
                x_test,
                y_train,
                y_test,
                run.params.learning_rate,
                run.params.batch_size,
                False
            )
        # evaluate the model
        train_mse, train_gme = model.evaluate_model(x_train, y_train)
        test_mse, test_gme = model.evaluate_model(x_test, y_test)
        # log the model weights
        weights_train.append(len(x_train))
        weights_test.append(len(x_test))
        train_mses.append(train_mse)
        train_gmses.append(train_gme)
        test_mses.append(test_mse)
        test_gmses.append(test_gme)
    train_mse = 0
    train_gmse = 0
    test_mse = 0
    test_gmse = 0
    for i in range(30):
        train_mse += weights_train[i] * train_mses[i]
        train_gmse += weights_train[i] * train_gmses[i]
        test_mse += weights_test[i] * test_mses[i]
        test_gmse += weights_test[i] * test_gmses[i]
    train_mse /= sum(weights_train)
    train_gmse /= sum(weights_train)
    test_mse /= sum(weights_test)
    test_gmse /= sum(weights_test)
    run.log_metric("train gMSE", train_gmse)
    run.log_metric("train MSE", train_mse)
    run.log_metric("test gMSE", test_gmse)
    run.log_metric("test MSE", test_mse)
    return


if __name__ == "__main__":
    CONV_INPUT_LENGTH = 288
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Specify an experiment name')
    args = parser.parse_args()
    name = args.name
    if not name:
        name=''

    data = load_data(0.8, 0.0)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    experiment = sigopt.create_experiment(
        name=f"Baseline_3_{name}",
        type="offline",
        parameters=[
            dict(
                name="activation",
                type="categorical",
                categorical_values=["relu", "tanh"],
            ),
            dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.5)),
            dict(
                name="learning_rate", type="double", bounds=dict(min=0.00001, max=0.01)
            ),
            dict(name="num_epochs", type="int", bounds=dict(min=1, max=10)),
            dict(name="batch_size", type="int", bounds=dict(min=32, max=64)),
            dict(name="filter_1", type="int", bounds=dict(min=1, max=10)),
            dict(name="kernel_1", type="int", bounds=dict(min=5, max=10)),
            dict(name="stride_1", type="int", bounds=dict(min=1, max=2)),
            dict(name="pool_size_1", type="int", bounds=dict(min=1, max=3)),
            dict(name="pool_stride_1", type="int", bounds=dict(min=1, max=2)),
            dict(name="filter_2", type="int", bounds=dict(min=1, max=5)),
            dict(name="kernel_2", type="int", bounds=dict(min=2, max=5)),
            dict(name="stride_2", type="int", bounds=dict(min=1, max=2)),
            dict(name="pool_size_2", type="int", bounds=dict(min=1, max=2)),
            dict(name="pool_stride_2", type="int", bounds=dict(min=1, max=2)),
        ],
        metrics=[dict(name="test gMSE", strategy="optimize", objective="minimize")],
        linear_constraints=[
            dict(
                type="greater_than",
                threshold=0,
                terms=[
                    dict(name="kernel_1", weight=1),
                    dict(name="stride_1", weight=-1),
                ],
            ),
            dict(
                type="greater_than",
                threshold=0,
                terms=[
                    dict(name="kernel_2", weight=1),
                    dict(name="stride_2", weight=-1),
                ],
            ),
            dict(
                type="greater_than",
                threshold=0,
                terms=[
                    dict(name="pool_size_1", weight=1),
                    dict(name="pool_stride_1", weight=-1),
                ],
            ),
            dict(
                type="greater_than",
                threshold=0,
                terms=[
                    dict(name="pool_size_1", weight=1),
                    dict(name="pool_stride_1", weight=-1),
                ],
            ),
        ],
        parallel_bandwidth=3,
        budget=100,
    )
    for run in experiment.loop():
        with run:
            run.log_metadata("commit", sha)
            run.log_metadata("GPUs available", tf.config.list_physical_devices("GPU"))
            load_data_train_model(run, data, CONV_INPUT_LENGTH)
