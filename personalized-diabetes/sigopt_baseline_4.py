# Python file for running baseline 4 (individualization, self-supervised data) using sigopt
import pandas as pd
import sigopt_functions as sf
import sigopt
import git
import os
import tensorflow as tf


os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"
#os.environ['CUDA_VISIBLE_DEVICES'] ="0"
DATASET = 'basic_0.csv'
DATASET_SELF = 'self_0.csv'

def load_data(split:float, data_missingness:float):
    df_basic = pd.read_csv(DATASET)
    print('Basic data read')
    df_self = pd.read_csv(DATASET_SELF)
    print('Self data read')
    # delete a fraction of the df rows according to data_missingness
    df_basic = sf.apply_data_missingness(df_basic, data_missingness)
    X_train, X_test, Y_train, Y_test = \
        sf.get_train_test_split_search(df_basic, split, False)
    X_train_self, X_test_self, Y_train_self, Y_test_self = \
        sf.get_train_test_split_search(df_self, split, True)

    return X_train, X_test, Y_train, Y_test, X_train_self, X_test_self, Y_train_self, Y_test_self

def load_data_train_model(run, data, CONV_INPUT_LENGTH):
    run.log_dataset(name=DATASET)
    X_train, X_test, Y_train, Y_test, X_train_self, X_test_self, Y_train_self, Y_test_self = data
    weights = []
    train_mses = []
    train_gmses = []
    test_mses = []
    test_gmses = []
    run.log_model("Baseline 4")
    run.log_metadata("sgd optimizer", "adam")
    for i in range(1, 31):
        # create the model
        model = \
            sf.GlucoseModel(CONV_INPUT_LENGTH, True, run)
        x_train = X_train_self[X_train_self['DeidentID'] == i]
        x_test = X_test_self[X_test_self['DeidentID'] == i]
        y_train = Y_train_self[Y_train_self['DeidentID'] == i]
        y_test = Y_test_self[Y_test_self['DeidentID'] == i]
        # self-supervised training
        model.train_model(run.params.num_epochs_1, x_train, x_test, y_train, y_test,
                          run.params.learning_rate_1, run.params.batch_size_1)
        # individualization
        model.activate_finetune_mode()
        x_train = X_train[X_train['DeidentID'] == i]
        x_test = X_test[X_test['DeidentID'] == i]
        y_train = Y_train[Y_train['DeidentID'] == i]
        y_test = Y_test[Y_test['DeidentID'] == i]
        model.train_model(run.params.num_epochs_2, x_train, x_test, y_train, y_test,
                            run.params.learning_rate_2, run.params.batch_size_2)
        # evaluate the model
        train_mse, train_gme = model.evaluate_model(x_train, y_train)
        test_mse, test_gme = model.evaluate_model(x_test, y_test)
        # log the model weights
        weights.append(len(x_train))
        train_mses.append(train_mse)
        train_gmses.append(train_gme)
        test_mses.append(test_mse)
        test_gmses.append(test_gme)
    train_mse = 0
    train_gmse = 0
    test_mse = 0
    test_gmse = 0
    for i in range(30):
        train_mse += weights[i] * train_mses[i]
        train_gmse += weights[i] * train_gmses[i]
        test_mse += weights[i] * test_mses[i]
        test_gmse += weights[i] * test_gmses[i]
    train_mse /= sum(weights)
    train_gmse /= sum(weights)
    test_mse /= sum(weights)
    test_gmse /= sum(weights)
    run.log_metric("train gMSE", train_gmse)
    run.log_metric("train MSE", train_mse)
    run.log_metric("test gMSE", test_gmse)
    run.log_metric("test MSE", test_mse)
    return


if __name__ == '__main__':
    CONV_INPUT_LENGTH = 288
    data = load_data(0.8, 0.0)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    experiment = sigopt.create_experiment(
        name="Baseline_1_experiment",
        type="offline",
        parameters=[
            dict(name="activation", type="categorical", categorical_values=["relu", "tanh"]),
            dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.5)),
            dict(name="learning_rate_1", type="double", bounds=dict(min=0.00001, max=0.01)),
            dict(name='num_epochs_1', type="int", bounds=dict(min=1, max = 15)),
            dict(name='batch_size_1', type = "int", bounds=dict(min=2, max=32)),
            dict(name="learning_rate_2", type="double", bounds=dict(min=0.00001, max=0.01)),
            dict(name='num_epochs_2', type="int", bounds=dict(min=1, max=15)),
            dict(name='batch_size_2', type="int", bounds=dict(min=2, max=32)),
            dict(name='filter_1', type = "int", bounds=dict(min=1, max=10)),
            dict(name='kernel_1', type="int", bounds=dict(min=5, max=10)),
            dict(name='stride_1', type="int", bounds=dict(min=1, max=2)),
            dict(name='pool_size_1', type="int", bounds=dict(min=1, max=3)),
            dict(name='pool_stride_1', type="int", bounds=dict(min=1, max=2)),
            dict(name='filter_2', type="int", bounds=dict(min=1, max=5)),
            dict(name='kernel_2', type="int", bounds=dict(min=1, max=5)),
            dict(name='stride_2', type="int", bounds=dict(min=1, max=2)),
            dict(name='pool_size_2', type="int", bounds=dict(min=1, max=2)),
            dict(name='pool_stride_2', type="int", bounds=dict(min=1, max=2)),
        ],
        metrics=[dict(name="test gMSE", strategy="optimize", objective="minimize")],
        linear_constraints=[
            dict(type='greater_than', threshold=0, terms=[
                dict(name='kernel_1', weight=1),
                dict(name='stride_1', weight=-1)
            ]),
            dict(type='greater_than', threshold=0, terms=[
                dict(name='kernel_2', weight=1),
                dict(name='stride_2', weight=-1)
            ]),
            dict(type='greater_than', threshold=0, terms=[
                dict(name='pool_size_1', weight=1),
                dict(name='pool_stride_1', weight=-1)
            ]),
            dict(type='greater_than', threshold=0, terms=[
                dict(name='pool_size_1', weight=1),
                dict(name='pool_stride_1', weight=-1)
            ])
        ],
        parallel_bandwidth=1,
        budget=1000,
    )
    for run in experiment.loop():
        with run:
            run.log_metadata('commit', sha)
            run.log_metadata('GPUs available', tf.config.list_physical_devices('GPU'))
            load_data_train_model(run,data, CONV_INPUT_LENGTH)