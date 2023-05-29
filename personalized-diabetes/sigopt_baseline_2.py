# Python file for running baseline 2 (no individualization, self-supervised data) using sigopt
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
DATASET_SELF = "self_0.csv"


def load_data(split: float, data_missingness: float):
    # read in df_self but only read in every 4th row
    df_self = pd.read_csv(DATASET_SELF, skiprows=lambda i: i % 4 != 0)

    #sample to 1000 random rows
    df_self = df_self.sample(n=1000, random_state=1)
    print(f"Self supervised data read with shape {df_self.shape}")
    df_basic = pd.read_csv(DATASET, skiprows=lambda i: i % 2 != 0)
    df_basic = df_basic.sample(n=1000, random_state=1)
    print(f"Basic data read with shape {df_basic.shape}")

    # delete a fraction of the df rows according to data_missingness
    df_basic = sf.apply_data_missingness(df_basic, data_missingness)

    X_train, X_test, Y_train, Y_test = sf.get_train_test_split_search(
        df_basic, split, False
    )
    (
        X_train_self,
        X_test_self,
        Y_train_self,
        Y_test_self,
    ) = sf.get_train_test_split_search(df_self, split, True)
    X_train.drop(columns=["DeidentID"], inplace=True)
    X_test.drop(columns=["DeidentID"], inplace=True)
    Y_train.drop(columns=["DeidentID"], inplace=True)
    Y_test.drop(columns=["DeidentID"], inplace=True)
    X_train_self.drop(columns=["DeidentID"], inplace=True)
    X_test_self.drop(columns=["DeidentID"], inplace=True)
    Y_train_self.drop(columns=["DeidentID"], inplace=True)
    Y_test_self.drop(columns=["DeidentID"], inplace=True)
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


def load_data_train_model(run, data, CONV_INPUT_LENGTH):
    run.log_dataset(name=DATASET)
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

    # create the model
    with tf.device("/device:GPU:0"):
        model = sf.GlucoseModel(CONV_INPUT_LENGTH, True, run)
    run.log_model("Baseline 2")
    run.log_metadata("sgd optimizer", "adam")
    # train the model for self_supervised
    with tf.device("/device:GPU:0"):
        model.train_model(
            run.params.num_epochs_1,
            X_train_self,
            X_test_self,
            Y_train_self,
            Y_test_self,
            run.params.learning_rate_1,
            run.params.batch_size_1, True
        )
        # supervised training
        model.activate_finetune_mode()
        model.train_model(
            run.params.num_epochs_2,
            X_train,
            X_test,
            Y_train,
            Y_test,
            run.params.learning_rate_2,
            run.params.batch_size_2,
            False
        )
        train_loss, train_mse = model.evaluate_model(X_train, Y_train)
        test_loss, test_mse = model.evaluate_model(X_test, Y_test)
        print(test_loss)
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
        name=f"Baseline_2_{name}",
        type="offline",
        parameters=[
            dict(
                name="activation",
                type="categorical",
                categorical_values=["relu", "tanh"],
            ),
            dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.5)),
            dict(
                name="learning_rate_1",
                type="double",
                bounds=dict(min=0.00001, max=0.01),
            ),
            dict(
                name="learning_rate_2",
                type="double",
                bounds=dict(min=0.00001, max=0.01),
            ),
            dict(name="num_epochs_1", type="int", bounds=dict(min=1, max=10)),
            dict(name="num_epochs_2", type="int", bounds=dict(min=1, max=10)),
            dict(name="batch_size_1", type="int", bounds=dict(min=32, max=64)),
            dict(name="batch_size_2", type="int", bounds=dict(min=2, max=32)),
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
        parallel_bandwidth=1,
        budget=100,
    )
    for run in experiment.loop():
        with run:
            run.log_metadata("commit", sha)
            run.log_metadata("GPUs available", tf.config.list_physical_devices("GPU"))
            load_data_train_model(run, data, CONV_INPUT_LENGTH)
