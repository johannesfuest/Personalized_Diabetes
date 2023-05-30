# Python file for running baseline 1 (no individualization, no self-supervised data) using sigopt
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
    df_basic = df_basic.iloc[::2, :]

    X_train, X_test, Y_train, Y_test = sf.get_train_test_split_search(
        df_basic, split, False
    )
    X_train.drop(columns=["DeidentID"], inplace=True)
    X_test.drop(columns=["DeidentID"], inplace=True)
    Y_train.drop(columns=["DeidentID"], inplace=True)
    Y_test.drop(columns=["DeidentID"], inplace=True)
    return X_train, X_test, Y_train, Y_test


def load_data_train_model(run, data, CONV_INPUT_LENGTH):
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
            run.params.batch_size,
        )
    run.log_metadata("sgd optimizer", "adam")
    train_loss, train_mse = model.evaluate_model(X_train, Y_train)
    test_loss, test_mse = model.evaluate_model(X_test, Y_test)
    # log performance metrics
    run.log_metric("train gMSE", train_loss)
    run.log_metric("train MSE", train_mse)
    run.log_metric("test gMSE", test_loss)
    run.log_metric("test MSE", test_mse)


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
        name=f"Baseline_1_{name}",
        type="offline",
        parameters=[
            dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.2)),
            dict(
                name="learning_rate", type="double", bounds=dict(min=0.0008, max=0.0015)
            ),
            dict(name="num_epochs", type="int", bounds=dict(min=8, max=12)),
            dict(name="batch_size", type="categorical", categorical_values=['32', '64']),
            dict(name="filter_1", type="int", bounds=dict(min=2, max=4)),
            dict(name="kernel_1", type="int", bounds=dict(min=5, max=7)),
            dict(name="stride_1", type="int", bounds=dict(min=1, max=2)),
            dict(name="pool_size_1", type="int", bounds=dict(min=1, max=3)),
            dict(name="pool_stride_1", type="int", bounds=dict(min=1, max=2)),
            dict(name="filter_2", type="int", bounds=dict(min=5, max=7)),
            dict(name="kernel_2", type="int", bounds=dict(min=4, max=6)),
            dict(name="stride_2", type="int", bounds=dict(min=1, max=2)),
            dict(name="pool_size_2", type="int", bounds=dict(min=5, max=6)),
            dict(name="pool_stride_2", type="int", bounds=dict(min=3, max=5)),
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
        parallel_bandwidth=1,
        budget=100,
    )
    for run in experiment.loop():
        with run:
            run.log_metadata("commit", sha)
            run.log_metadata("GPUs available", tf.config.list_physical_devices("GPU"))
            load_data_train_model(run, data, CONV_INPUT_LENGTH)
