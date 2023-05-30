# Python file for running baseline 1 (individualization, self-supervised data, double pretraining) using sigopt
import pandas as pd
import sigopt_functions as sf
import sigopt
import git
import os
import tensorflow as tf
import argparse




# TODO: make GlucoseModel class inherit from Model class
#TODO: set up new AWS instance with new limit
#TODO: write experiment code for all models



os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"
DATASET = 'basic_0.csv'
DATASET_SELF = 'self_0.csv'

def load_data(split:float, data_missingness:float):

    df_basic = pd.read_csv(DATASET, skiprows=lambda i: i % 2 != 0)
    print('Basic data read')
    df_self = pd.read_csv(DATASET_SELF, skiprows=lambda i: i % 4 != 0)
    print('Self data read')
    df_basic = df_basic.sample(n=1000, random_state=1)

    df_self = df_self.sample(n=1000, random_state=1)
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
    weights_train = []
    weights_test = []
    train_mses = []
    train_gmses = []
    test_mses = []
    test_gmses = []
    run.log_model("Final Model")
    run.log_metadata("sgd optimizer", "adam")
    x_train_temp = X_train
    x_test_temp = X_test
    y_train_temp = Y_train
    y_test_temp = Y_test
    x_train_temp = x_train_temp.drop(columns = ['DeidentID'])
    x_test_temp = x_test_temp.drop(columns = ['DeidentID'])
    y_train_temp = y_train_temp.drop(columns = ['DeidentID'])
    y_test_temp = y_test_temp.drop(columns = ['DeidentID'])
    with tf.device('/device:GPU:0'):
        base_model = \
                sf.GlucoseModel(CONV_INPUT_LENGTH, True, run)
        # pretrain the model on all patient data
        base_model.train_model(run.params.num_epochs_0,x_train_temp, x_test_temp, y_train_temp, y_test_temp,
                          run.params.learning_rate_0, run.params.batch_size_0, True)
    for i in range(1, 31):
        with tf.device('/device:GPU:0'):
            #clone the base model
            glucose_temp = sf.GlucoseModel(CONV_INPUT_LENGTH, True, run)
            glucose_temp.model.set_weights(base_model.model.get_weights())

        # create the model
        x_train = X_train_self[X_train_self['DeidentID'] == i]
        x_test = X_test_self[X_test_self['DeidentID'] == i]
        y_train = Y_train_self[Y_train_self['DeidentID'] == i]
        y_test = Y_test_self[Y_test_self['DeidentID'] == i]
        x_train = x_train.drop(columns=['DeidentID'])
        x_test = x_test.drop(columns=['DeidentID'])
        y_train = y_train.drop(columns=['DeidentID'])
        y_test = y_test.drop(columns=['DeidentID'])
        # self-supervised training
        with tf.device('/device:GPU:0'):
            glucose_temp.train_model(run.params.num_epochs_1, x_train, x_test, y_train, y_test,
                              run.params.learning_rate_1, run.params.batch_size_1, True)
            # individualization
            glucose_temp.activate_finetune_mode()
        x_train = X_train[X_train['DeidentID'] == i]
        x_test = X_test[X_test['DeidentID'] == i]
        y_train = Y_train[Y_train['DeidentID'] == i]
        y_test = Y_test[Y_test['DeidentID'] == i]
        x_train = x_train.drop(columns=['DeidentID'])
        x_test = x_test.drop(columns=['DeidentID'])
        y_train = y_train.drop(columns=['DeidentID'])
        y_test = y_test.drop(columns=['DeidentID'])
        with tf.device('/device:GPU:0'):
            glucose_temp.train_model(run.params.num_epochs_2, x_train, x_test, y_train, y_test,
                                run.params.learning_rate_2, run.params.batch_size_2, False)
            # evaluate the model
            train_mse, train_gme = glucose_temp.evaluate_model(x_train, y_train)
            test_mse, test_gme = glucose_temp.evaluate_model(x_test, y_test)
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


if __name__ == '__main__':
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
        name=f"Final_model_local_debug_{name}",
        type="offline",
        parameters=[
            dict(name="activation", type="categorical", categorical_values=["relu", "tanh"]),
            dict(name="dropout_rate", type="double", bounds=dict(min=0.0, max=0.5)),
            dict(name="learning_rate_0", type="double", bounds=dict(min=0.00001, max=0.01)),
            dict(name='num_epochs_0', type="int", bounds=dict(min=1, max=10)),
            dict(name='batch_size_0', type="int", bounds=dict(min=32, max=64)),
            dict(name="learning_rate_1", type="double", bounds=dict(min=0.00001, max=0.01)),
            dict(name='num_epochs_1', type="int", bounds=dict(min=1, max = 10)),
            dict(name='batch_size_1', type = "int", bounds=dict(min=32, max=64)),
            dict(name="learning_rate_2", type="double", bounds=dict(min=0.00001, max=0.01)),
            dict(name='num_epochs_2', type="int", bounds=dict(min=1, max=10)),
            dict(name='batch_size_2', type="int", bounds=dict(min=32, max=64)),
            dict(name='filter_1', type = "int", bounds=dict(min=1, max=10)),
            dict(name='kernel_1', type="int", bounds=dict(min=5, max=10)),
            dict(name='stride_1', type="int", bounds=dict(min=1, max=2)),
            dict(name='pool_size_1', type="int", bounds=dict(min=1, max=3)),
            dict(name='pool_stride_1', type="int", bounds=dict(min=1, max=2)),
            dict(name='filter_2', type="int", bounds=dict(min=1, max=5)),
            dict(name='kernel_2', type="int", bounds=dict(min=2, max=5)),
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
        budget=100,
    )
    for run in experiment.loop():
        with run:
            run.log_metadata('commit', sha)
            run.log_metadata('GPUs available', tf.config.list_physical_devices('GPU'))
            load_data_train_model(run,data, CONV_INPUT_LENGTH)