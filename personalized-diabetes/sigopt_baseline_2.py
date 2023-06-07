# Python file for running baseline 2 (no individualization, self-supervised data) using sigopt
import argparse
import git
import os
import pandas as pd
import sigopt_functions as sf
import sigopt
import tensorflow as tf


os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"
DATASET = "basic_0.csv"
DATASET_SELF = "self_sup_alt.csv"

def load_data(split: float, missingness_modulo: int):
    # read in df_self but only read in every 4th row
    df_self = pd.read_csv(DATASET_SELF)
    print(f"Self supervised data read with shape {df_self.shape}")
    df_basic = pd.read_csv(DATASET)
    print(f"Basic data read with shape {df_basic.shape}")

    # delete a fraction of the df rows according to data_missingness

    X_train, X_test, Y_train, Y_test = sf.get_train_test_split_search(
        df_basic, split, False
    )
    X_train, Y_train = sf.apply_data_missingness(X_train, Y_train, missingness_modulo)
    (
        X_train_self,
        X_test_self,
        Y_train_self,
        Y_test_self,
    ) = sf.get_train_test_split_search(df_self, split, True)
    Y_train.drop(columns=["DeidentID"], inplace=True)
    Y_test.drop(columns=["DeidentID"], inplace=True)
    Y_train_self.drop(columns=["DeidentID"], inplace=True)
    Y_test_self.drop(columns=["DeidentID"], inplace=True)
    X_train_self.drop(columns=["DeidentID"], inplace=True)
    X_test_self.drop(columns=["DeidentID"], inplace=True)

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


def load_data_train_model(run, data, CONV_INPUT_LENGTH, write_preds=False):
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

    x_train_ids = X_train['DeidentID']
    x_test_ids = X_test['DeidentID']
    X_train.drop(columns=['DeidentID'], inplace=True)
    X_test.drop(columns=['DeidentID'], inplace=True)


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
            int(run.params.batch_size), True
        )
        # supervised training
    model.activate_finetune_mode()
    with tf.device("/device:GPU:0"):
        model.train_model(
            run.params.num_epochs_2,
            X_train,
            X_test,
            Y_train,
            Y_test,
            run.params.learning_rate_2,
            int(run.params.batch_size),
            False
        )
        train_gmse, train_mse = model.evaluate_model(X_train, Y_train)
        test_gmse, test_mse = model.evaluate_model(X_test, Y_test)

    print(f'len(x_train){len(X_train)})')
    print(f'len(x_test){len(X_test)})')
    print(f'train_mse{train_mse})')
    print(f'train_gme{train_gmse})')
    print(f'test_mse{test_mse})')
    print(f'test_gme{test_gmse})')

    print('Y-TRAIN:')
    print(Y_train.describe())
    print('Y-HAT-TRAIN:')
    train_preds = pd.DataFrame(model.model.predict(X_train))
    train_preds['y'] = Y_train['CGM'].values
    train_preds['DeidentID'] = x_train_ids.values
    print(train_preds.describe())

    train_preds['run'] = run.id
    train_preds['experiment'] = run.experiment
    print('Y-TEST:')
    print(Y_test.describe())
    print('Y-HAT-TEST:')
    test_preds = pd.DataFrame(model.model.predict(X_test))
    test_preds['y'] = Y_test['CGM'].values
    test_preds['DeidentID'] = x_test_ids.values
    test_preds['run'] = run.id
    test_preds['experiment'] = run.experiment
    print(test_preds.describe())


    if write_preds:
        if not os.path.exists('preds'):
            os.mkdir('preds')
        train_preds.to_csv(os.path.join('preds', f'base_7_train_M{run.params.missingness_modulo}.csv'))
        test_preds.to_csv(os.path.join('preds', f'base_7_test_M{run.params.missingness_modulo}.csv'))

    # log performance metrics
    run.log_metric("train gMSE", train_gmse)
    run.log_metric("train MSE", train_mse)
    run.log_metric("test gMSE", test_gmse)
    run.log_metric("test MSE", test_mse)
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    CONV_INPUT_LENGTH = 288
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Specify an experiment name')
    parser.add_argument('--experiment', action='store_true', help='Enable experiment mode')
    args = parser.parse_args()
    name = args.name
    if not name:
        name=''

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if args.experiment:
        fixed_hyperparameters = {
        'dropout_rate':  0.0579,
        'learning_rate_1': 0.0013164,
        'learning_rate_2': 0.001362939,
        'num_epochs_1':    13,
        'num_epochs_2':  10,
        'batch_size':    64,
        'filter_1':      4,
        'kernel_1':      6,
        'stride_1':      2,
        'pool_size_1':   2,
        'pool_stride_1': 2,
        'filter_2':      7,
        'kernel_2':      5,
        'stride_2':      2,
        'pool_size_2':   6,
        'pool_stride_2': 5,
        }
        experiment = sigopt.create_experiment(
            name=f"Baseline_2_EXPERIMENT_{name}",
            type="grid",
            parameters=[
                dict(name="missingness_modulo", type="int", grid=[800, 1000])
            ],
            metrics=[dict(name="test gMSE", strategy="optimize", objective="minimize")],
            parallel_bandwidth=1,
            #budget=11,
        )

        for run in experiment.loop():
            with run:
                data = load_data(0.8, run.params.missingness_modulo)
                for parameter, value in fixed_hyperparameters.items():
                    run.params[parameter] = value
                    run.log_metadata(parameter, value)
                run.log_metadata("commit", sha)
                run.log_metadata("GPUs available", tf.config.list_physical_devices("GPU"))
                load_data_train_model(run, data, CONV_INPUT_LENGTH, write_preds=True)
    else:

        data = load_data(0.8, 1)
        experiment = sigopt.create_experiment(
            name=f"Baseline_2_{name}",
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
