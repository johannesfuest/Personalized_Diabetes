# Python file for running baseline 2 (no individualization, self-supervised data) using sigopt
import pandas as pd
import sigopt_functions as sf
import sigopt
import git
import os
import tensorflow as tf


os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"
DATASET = 'basic_0.csv'
DATASET_SELF = 'self_0.csv'


def load_data(split:float, data_missingness:float):
    df_basic = pd.read_csv(DATASET)
    print('Basic data read')
    df_self = pd.read_csv(DATASET_SELF)
    print('Self supervised data read')
    # delete a fraction of the df rows according to data_missingness
    df_basic = sf.apply_data_missingness(df_basic, data_missingness)

    X_train, X_test, Y_train, Y_test = \
        sf.get_train_test_split_search(df_basic, split, False)
    X_train_self, X_test_self, Y_train_self, Y_test_self = sf.get_train_test_split_search(df_self, split, True)
    return X_train, X_test, Y_train, Y_test, X_train_self, X_test_self, Y_train_self, Y_test_self

def load_data_train_model(run, data, CONV_INPUT_LENGTH):
    run.log_dataset(name=DATASET)
    X_train, X_test, Y_train, Y_test, X_train_self, X_test_self, Y_train_self, Y_test_self = data

    # create the model
    model = \
        sf.GlucoseModel(CONV_INPUT_LENGTH, True, run)
    run.log_model("Baseline 2")
    run.log_metadata("sgd optimizer", "adam")
    # train the model for self_supervised
    model.train_model(run.params.num_epochs_1, X_train_self, X_test_self, Y_train_self, Y_test_self,
                      run.params.learning_rate_1, run.params.batch_size_1)
    # supervised training
    model.activate_finetune_mode()
    model.train_model(run.params.num_epochs_2, X_train, X_test, Y_train, Y_test,
                      run.params.learning_rate_2, run.params.batch_size_2)
    train_loss, train_mse = model.evaluate_model(X_train, Y_train)
    test_loss, test_mse = model.evaluate_model(X_test, Y_test)
    # log performance metrics
    run.log_metric("train gMSE", train_loss)
    run.log_metric("train MSE", train_mse)
    run.log_metric("test gMSE", test_loss)
    run.log_metric("test MSE", test_mse)


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
            dict(name="learning_rate_2", type="double", bounds=dict(min=0.00001, max=0.01)),
            dict(name='num_epochs_1', type="int", bounds=dict(min=0, max = 50)),
            dict(name='num_epochs_2', type="int", bounds=dict(min=0, max=50)),
            dict(name='batch_size_1', type = "int", bounds=dict(min=8, max=256)),
            dict(name='batch_size_2', type="int", bounds=dict(min=8, max=256)),
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