# Python file for running baseline 1 (individualization, self-supervised data, double pretraining) using sigopt
import argparse
import git
import numpy as np
import os
import pandas as pd
import sigopt_functions as sf
import sigopt
import tensorflow as tf


os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"
DATASET = 'basic_0.csv'
DATASET_SELF = 'self_sup_alt.csv'


def load_data(split:float, missingness_modulo:int):

    df_basic = pd.read_csv(DATASET)
    print('Basic data read')
    df_self = pd.read_csv(DATASET_SELF)
    print('Self data read')
    X_train, X_test, Y_train, Y_test = \
        sf.get_train_test_split_search(df_basic, split, False)
    
    X_train, Y_train = sf.apply_data_missingness(x_train=X_train, y_train=Y_train, missingness_modulo=missingness_modulo)
    X_train_self, X_test_self, Y_train_self, Y_test_self = \
        sf.get_train_test_split_search(df_self, split, True)
    return X_train, X_test, Y_train, Y_test, X_train_self, X_test_self, Y_train_self, Y_test_self

def load_data_train_model(run, data, CONV_INPUT_LENGTH, write_preds=False):
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

    print("Starting general self-supervised training")
    with tf.device('/device:GPU:0'):
        base_model = \
                sf.GlucoseModel(CONV_INPUT_LENGTH, True, run)
        # pretrain the model on all patient data
        base_model.train_model(run.params.num_epochs_1,X_train_self.drop(columns=['DeidentID']),
                               X_test_self.drop(columns=['DeidentID']), Y_train_self.drop(columns=['DeidentID']),
                                 Y_test_self.drop(columns=['DeidentID']), run.params.learning_rate_1,
                                    int(run.params.batch_size), True)
        print("Finished general self-supervised training")
        base_model.activate_finetune_mode()
        # General supervised training
        print("Starting general supervised training")
        base_model.train_model(run.params.num_epochs_2, X_train.drop(columns=['DeidentID']),
                               X_test.drop(columns=['DeidentID']), Y_train.drop(columns=['DeidentID']),
                               Y_test.drop(columns=['DeidentID']), run.params.learning_rate_2,
                               int(run.params.batch_size), False)
        print("Finished general supervised training")


    patients_to_exclude = [1, 9, 10, 12, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30]
    patients_to_include = [i for i in range(1, 31) if i not in patients_to_exclude]
    for i in patients_to_include:
        with tf.device('/device:GPU:0'):
            #clone the base model
            glucose_temp = sf.GlucoseModel(CONV_INPUT_LENGTH, True, run)
            glucose_temp.model.set_weights(base_model.model.get_weights())

        x_train = X_train[X_train['DeidentID'] == i]
        x_test = X_test[X_test['DeidentID'] == i]
        y_train = Y_train[Y_train['DeidentID'] == i]
        y_test = Y_test[Y_test['DeidentID'] == i]
        x_train = x_train.drop(columns=['DeidentID'])
        x_test = x_test.drop(columns=['DeidentID'])
        y_train = y_train.drop(columns=['DeidentID'])
        y_test = y_test.drop(columns=['DeidentID'])
        print("Starting individual supervised training for patient " + str(i))
        with tf.device('/device:GPU:0'):
            glucose_temp.train_model(run.params.num_epochs_2, x_train, x_test, y_train, y_test,
                                run.params.learning_rate_2, int(run.params.batch_size), False)
            # evaluate the model
        print("Finished individual supervised training for patient " + str(i))
         # evaluate the model
        train_gmse, train_mse = glucose_temp.evaluate_model(x_train, y_train)
        test_gmse, test_mse = glucose_temp.evaluate_model(x_test, y_test)

        print(f'len(x_train){len(x_train)})')
        print(f'len(x_test){len(x_test)})')
        print(f'train_mse{train_mse})')
        print(f'train_gme{train_gmse})')
        print(f'test_mse{test_mse})')
        print(f'test_gme{test_gmse})')

        print('Y-TRAIN:')
        print(y_train.describe())
        print('Y-HAT-TRAIN:')
        train_preds = pd.DataFrame(glucose_temp.model.predict(x_train))
        train_preds['y'] = y_train['CGM'].values
        print(train_preds.describe())

        train_preds['run'] = run.id
        train_preds['experiment'] = run.experiment
        print('Y-TEST:')
        print(y_test.describe())
        print('Y-HAT-TEST:')
        test_preds = pd.DataFrame(glucose_temp.model.predict(x_test))
        test_preds['y'] = y_test['CGM'].values
        test_preds['run'] = run.id
        test_preds['experiment'] = run.experiment
        print(test_preds.describe())

        if write_preds:
            if not os.path.exists('preds'):
                os.mkdir('preds')
            train_preds.to_csv(os.path.join('preds', f'base_9_train_M{run.params.missingness_modulo}_D{i}.csv'))
            test_preds.to_csv(os.path.join('preds', f'base_9_test_M{run.params.missingness_modulo}_D{i}.csv'))

        print('Y-TEST:')
        print(y_test.describe())
        print('Y-HAT-TEST:')
        print(test_preds.describe())
        # log the model weights
        weights_train.append(len(x_train))
        weights_test.append(len(x_test))
        train_mses.append(train_mse)
        train_gmses.append(train_gmse)
        test_mses.append(test_mse)
        test_gmses.append(test_gmse)

    train_gmses = np.clip(train_gmses, a_max=10000000, a_min=0)
    train_mses = np.clip(train_mses, a_max=10000000, a_min=0)
    test_gmses = np.clip(test_gmses, a_max=10000000, a_min=0)
    test_mses = np.clip(test_mses, a_max=10000000, a_min=0)
    run.log_metric('u train gMSE', np.mean(train_gmses))
    run.log_metric('u train MSE', np.mean(train_mses))
    run.log_metric('u test gMSE', np.mean(test_gmses))
    run.log_metric('u test gMSE', np.mean(test_mses))
    
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

if __name__ == '__main__':
    CONV_INPUT_LENGTH = 288
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Specify an experiment name')
    parser.add_argument('--experiment', action='store_true', help='Enable experiment mode')
    # set allow growth to true to avoid OOM errors
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    args = parser.parse_args()
    name = args.name
    if not name:
        name=''
        
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    if args.experiment:
        fixed_hyperparameters = {
        'dropout_rate':  0.04789191476471058,
        'learning_rate_1': 0.0005718050423761871,
        'learning_rate_2': 0.0010113473984109185,
        'num_epochs_1':    11,
        'num_epochs_2':  10,
        'batch_size':    16,
        'filter_1':      3,
        'kernel_1':      6,
        'stride_1':      2,
        'pool_size_1':   3,
        'pool_stride_1': 2,
        'filter_2':      7,
        'kernel_2':      6,
        'stride_2':      2,
        'pool_size_2':   6,
        'pool_stride_2': 4,
        }
        experiment = sigopt.create_experiment(
            name=f"FINAL_EXPERIMENT_{name}",
            type="grid",
            parameters=[
                dict(name="missingness_modulo", type="int", grid=[10, 20, 50, 100, 200, 400, 1000])
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
            name=f"Final_model_search_{name}",
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