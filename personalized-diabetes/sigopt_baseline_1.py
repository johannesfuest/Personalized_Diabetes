# Python file for running baseline 1 (no individualization, no self-supervised data) using sigopt
import pandas as pd
import sigopt_functions as sf
import sigopt
import git
import os
import tensorflow as tf


os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATASET = 'basic_0.csv'


def load_data(split:float, data_missingness:float):
    df_basic = pd.read_csv(DATASET)
    print('data read')
    # delete a fraction of the df rows according to data_missingness
    df_basic = sf.apply_data_missingness(df_basic, data_missingness)

    X_train, X_test, Y_train, Y_test = \
        sf.get_train_test_split_search(df_basic, split, False)
    return X_train, X_test, Y_train, Y_test

def load_data_train_model(run, data, CONV_INPUT_LENGTH):
    run.log_dataset(name=DATASET)
    X_train, X_test, Y_train, Y_test = data

    # create the model
    model = \
        sf.GlucoseModel(CONV_INPUT_LENGTH, False, run)
    run.log_model("Baseline 1")
    # train the model
    model.train_model(run.params.num_epochs, X_train, X_test, Y_train, Y_test,
                      run.params.learning_rate, run.params.batch_size)
    run.log_metadata("sgd optimizer", "adam")
    train_loss, train_mse = model.evaluate_model(X_train, Y_train)
    test_loss, test_mse = model.evaluate_model(X_test, Y_test)
    # log performance metrics
    run.log_metric("train gMSE", train_loss)
    run.log_metric("train MSE", train_mse)
    run.log_metric("test gMSE", test_loss)
    run.log_metric("test MSE", test_mse)


if __name__ == '__main__':
    # stop tensorflow from using up over 50% of ram on the gpu
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


    CONV_INPUT_LENGTH = 288
    data = load_data(0.8, 0.0)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    experiment = sigopt.create_experiment(
        name="Baseline_1_debugging",
        type="offline",
        parameters=[
            dict(name="activation", type="categorical", categorical_values=["relu", "tanh"]),
            dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.5)),
            dict(name="learning_rate", type="double", bounds=dict(min=0.00001, max=0.01)),
            dict(name='num_epochs', type="int", bounds=dict(min=1, max = 20)),
            dict(name='batch_size', type = "int", bounds=dict(min=2, max=4)),
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
