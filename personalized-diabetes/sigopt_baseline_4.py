# Python file for running baseline 4 (individualization, self-supervised data) using sigopt
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
    #df_self = pd.read_csv(DATASET_SELF, skiprows=lambda i: i % 4 != 0)
    df_self = pd.read_csv(DATASET_SELF)
    print("Self supervised data read")
    df_basic = pd.read_csv(DATASET)
    # df_basic = pd.read_csv(DATASET, skiprows=lambda i: i % 2 != 0)
    print("Basic data read")
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
    weights_train = []
    weights_test = []
    train_mses = []
    train_gmses = []
    test_mses = []
    test_gmses = []
    run.log_model("Baseline 4")
    run.log_metadata("sgd optimizer", "adam")
    for i in range(1, 31):
        # create the model
        with tf.device("/GPU:0"):
            model = sf.GlucoseModel(CONV_INPUT_LENGTH, True, run)
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
                run.params.num_epochs_1,
                x_train,
                x_test,
                y_train,
                y_test,
                run.params.learning_rate_1,
                int(run.params.batch_size),
                True,
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
                run.params.num_epochs_2,
                x_train,
                x_test,
                y_train,
                y_test,
                run.params.learning_rate_2,
                int(run.params.batch_size),
                False
            )
            # evaluate the model
            train_gmse, train_mse = model.evaluate_model(x_train, y_train)
            test_gmse, test_mse = model.evaluate_model(x_test, y_test)
            preds_test = model.model.predict(x_test)
            #print max label
            print("Max labels: ", y_test.head())
            #print max prediction
            print("Max prediction: ", max(preds_test))
            print("Length preds: ", len(preds_test))
            print("Length y_test: ", len(y_test))
            print("Train MSE: ", train_mse)
            print("Train gMSE: ", train_gmse)
            print("Test MSE: ", test_mse)
            print("Test gMSE: ", test_gmse)


        print(len(x_train))
        print(len(x_test))
        # cap all the metrics at 10000000
        train_mse = min(train_mse, 10000000)
        train_gmse = min(train_gmse, 10000000)
        test_mse = min(test_mse, 10000000)
        test_gmse = min(test_gmse, 10000000)
        # log the model weights
        weights_train.append(len(x_train))
        weights_test.append(len(x_test))
        train_mses.append(train_mse)
        train_gmses.append(train_gmse)
        test_mses.append(test_mse)
        test_gmses.append(test_gmse)
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
    tf.keras.backend.clear_session()
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
        name=f"Baseline_4_{name}",
        type="offline",
        parameters=[dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.2)),
            dict(
                name="learning_rate_1", type="double", bounds=dict(min=0.0001, max=0.002)
            ),
            dict(name="learning_rate_2", type="double", bounds=dict(min=0.0008, max=0.0015)),
            dict(name="num_epochs_1", type="int", bounds=dict(min=5, max=15)),
            dict(name="num_epochs_2", type="int", bounds=dict(min=8, max=12)),
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
