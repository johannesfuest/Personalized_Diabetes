import sys
sys.path.append('..')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
import tensorflow as tf
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_rows', 1000)
import warnings
import sys
import random
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class GlucoseLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        pass

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, CONV_INPUT_LENGTH: int):
        super(ConvLayer, self).__init__()
        self.CONV_INPUT_LENGTH = CONV_INPUT_LENGTH
        self.conv1 = tfl.Conv1D(filters=3, kernel_size=5, strides=1, padding='valid')
        self.pool1 = tfl.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.conv2 = tfl.Conv1D(filters=6, kernel_size=5, strides=1, padding='valid')
        self.pool2 = tfl.MaxPool1D(pool_size=6, strides=4, padding='valid')
        self.flatten = tfl.Flatten()

    def build(self, input_shape):
        pass

    def call(self, input):
        assert input.shape[1] == self.CONV_INPUT_LENGTH
        # 1st CONV
        conv1_out = self.conv1(input)
        # Max Pool
        pool1_out = self.pool1(conv1_out)
        # 2nd CONV
        conv2_out = self.conv2(pool1_out)
        # Max Pool
        pool2_out = self.pool2(conv2_out)
        # Now flatten the Matrix into a 1D vector (shape 1x204)
        flatten_out = self.flatten(pool2_out)
        return flatten_out


def get_model(CONV_INPUT_LENGTH: int, ACTIVATION_FUNCTION: str, self_sup: bool):
    # define the input with specified shape
    # Don't specify batch-size to make pipeline more robust.
    input = tf.keras.Input(shape=(CONV_INPUT_LENGTH * 4, 1), batch_size=None)
    # Calculate batchsize of current run.
    batch_size = tf.shape(input)[0]

    # Slice the data into four equally sized 1D-chunks for CNN.
    diabetes_input = tf.slice(input, begin=[0, CONV_INPUT_LENGTH * 0, 0], size=[batch_size, CONV_INPUT_LENGTH, 1],
                              name='diabetes_input')
    meal_input = tf.slice(input, begin=[0, CONV_INPUT_LENGTH * 1, 0], size=[batch_size, CONV_INPUT_LENGTH, 1])
    smbg_input = tf.slice(input, begin=[0, CONV_INPUT_LENGTH * 2, 0], size=[batch_size, CONV_INPUT_LENGTH, 1])
    excercise_input = tf.slice(input, begin=[0, CONV_INPUT_LENGTH * 3, 0], size=[batch_size, CONV_INPUT_LENGTH, 1])

    # Create the four custom conv-layers
    diabetes_conv = ConvLayer(CONV_INPUT_LENGTH)(diabetes_input)
    meal_conv = ConvLayer(CONV_INPUT_LENGTH)(meal_input)
    smbg_conv = ConvLayer(CONV_INPUT_LENGTH)(smbg_input)
    excercise_conv = ConvLayer(CONV_INPUT_LENGTH)(excercise_input)

    # Concat the result of conv-layers
    post_conv = tf.concat([diabetes_conv, meal_conv, smbg_conv, excercise_conv], axis=1, name='post_conv')

    # Sanity check: Make sure that the shapes are as expected.
    assert post_conv.shape[1] == 204 * 4, 'Shape mismatch after conv layers'

    # Now fully connect layers
    # Use multiples of two as recommended in class.
    FC1 = tfl.Dense(units=512, activation=ACTIVATION_FUNCTION)(post_conv)
    FC2 = tfl.Dense(units=256, activation=ACTIVATION_FUNCTION)(FC1)
    FC3 = tfl.Dense(units=128, activation=ACTIVATION_FUNCTION)(FC2)
    FC4 = tfl.Dense(units=64, activation=ACTIVATION_FUNCTION)(FC3)
    # The output does NOT have an activation (regression task)
    if self_sup:
        output = tfl.Dense(units=4*CONV_INPUT_LENGTH, activation=None)(FC4)
    else:
        output = tfl.Dense(units=1, activation=None)(FC4)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def get_finetune_model(model):
    # Create new output layer which uses output of before-last layer as input
    output = tfl.Dense(units=1, activation=None)(model.layers[-2].output)
    # Create new model
    finetune_model = tf.keras.Model(inputs=model.input, outputs=output)
    return finetune_model


def get_train_test_split(df, TRAIN_TEST_SPLIT: float, self_sup: bool):
    """
    A function that takes a dataframe and returns a temporal train and test split using the given argument to determine
    split.
    :param df: the dataframe to be split
    :param TRAIN_TEST_SPLIT: the split to be used
    :param self_sup: Boolean indicating whether dataframe is self-supervised or not
    :return: X_train, X_test, Y_train, Y_test according to the given split
    """
    df = df.sort_values('LocalDtTm')
    # split into train and test
    train_length = int(TRAIN_TEST_SPLIT * df.shape[0])
    train = df.iloc[:train_length, :]
    test = df.iloc[train_length:, :]
    assert test.shape[0] + train.shape[0] == df.shape[0], 'Train-Test shapes don not add up.'
    if self_sup:
        X_train = train.drop(columns=['LocalDtTm', 'CGM'])
        Y_train = train.drop(columns=['LocalDtTm', 'CGM'])
        X_test = test.drop(columns=['LocalDtTm', 'CGM'])
        Y_test = test.drop(columns=['LocalDtTm', 'CGM'])
        for i in range(1, 289):
            X_train = X_train.drop(columns=[f'insulin {i} target', f'mealsize {i} target', f'carbs {i} target',
                                             f'exercise {i} target'])
            Y_train = Y_train.drop(columns=[f'insulin {i}', f'mealsize {i}', f'carbs {i}', f'exercise {i}'])
            X_test = X_test.drop(columns=[f'insulin {i} target', f'mealsize {i} target', f'carbs {i} target',
                                            f'exercise {i} target'])
            Y_test = Y_test.drop(columns=[f'insulin {i}', f'mealsize {i}', f'carbs {i}', f'exercise {i}'])
    else:
        X_train = train.drop(columns=['LocalDtTm', 'CGM'])
        Y_train = train[['CGM']]
        X_test = test.drop(columns=['LocalDtTm', 'CGM'])
        Y_test = test[['CGM']]
    return X_train, X_test, Y_train, Y_test

def get_train_test_split_all(df, TRAIN_TEST_SPLIT:float, self_sup:bool):
    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()
    for i in range(1,31):
        X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = get_train_test_split(df[df['DeidentID'] == i], TRAIN_TEST_SPLIT, self_sup)
        X_train = pd.concat([X_train, X_train_temp])
        Y_train = pd.concat([Y_train, Y_train_temp])
        X_test = pd.concat([X_test, X_test_temp])
        Y_test = pd.concat([Y_test, Y_test_temp])
    X_train.drop(columns=['DeidentID'], inplace=True)
    X_test.drop(columns=['DeidentID'], inplace=True)
    return X_train, X_test, Y_train, Y_test

def plot_predictions(preds, labels, patient: int, self_sup: bool, test: bool, indiv: bool):
    """
    A function that plots the predictions of the model.
    :param preds: predictions made by model
    :param labels: true data labels
    :param patient: patient number (0) for all patients
    :param self_sup: bool indicating whether model is self-supervised or not
    :param test: bool indicating whether test results or train results
    :param indiv: bool indicating whether individual patient or all patients
    :return: saves and displays a plot of the results
    """
    plt.scatter(labels, preds)
    plt.xlabel('True Blood Glucose Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    if indiv:
        if test:
            if self_sup:
                plt.title(f'Patient {patient} Final Model Test Results')
                # save plot
                plt.savefig(f'fina_testl_{patient}.png')
            else:
                plt.title(f'Patient {patient} Baseline 2 Test Results')
                # save plot
                plt.savefig(f'baseline2_test_{patient}.png')
        else:
            if self_sup:
                plt.title(f'Patient {patient} Final Model Train Results')
                # save plot
                plt.savefig(f'final_train_{patient}.png')
            else:
                plt.title(f'Patient {patient} Baseline 2 Train Results')
                # save plot
                plt.savefig(f'{patient}baseline2_train_{patient}.png')
    else:
        if test:
            if self_sup:
                plt.title(f'All Patients Baseline 3 Test Results')
                # save plot
                plt.savefig(f'all_baseline3_test.png')
            else:
                plt.title(f'All Patients Baseline 1 Test Results')
                # save plot
                plt.savefig(f'all_baseline2_test.png')
    plt.show()
    return

def train_model(model, epochs, X_train, X_test, Y_train, Y_test, lr, batch_size):
    """
    A function that trains the given model on the given data.
    :param model:
    :param epochs:
    :param train:
    :param test:
    :param lr:
    :param batch_size:
    :param mode: self for self-supervised, basic for basic
    :return:
    """
    # Create optimizer (Adam with specified learning rate - use default parameters otherwise. )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # TODO add optional gMSE loss function
    # Compile model - use mse for now.
    model.compile(optimizer=adam_optimizer,
                       loss='mse',
                       metrics=['mse'])
    # Create train and test datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)
    # Let's run this!
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=0)
    return model

def run_experiment(split: float = 0.8, lr: float = 0.001, epochs: int = 10, batch_size: int = 32, self_sup: bool = True,
                   activation: str = 'relu', indiv: bool = True, missingness: float = 0.0, dev: bool = False):
    """
    A function that runs an experiment with the given parameters.
    :param split: How much of the data should be used for training.
    Default value 0.8 => first 80% of time series used for training
    :param lr: The learning rate to be used by the adam optimizer.
    :param epochs: Epochs used for training. #TODO: alter between pretraining and finetuning?
    :param batch_size: The batch size to use in minibatch gradient descent.
    :param self_sup: Bool indicating whether to use self-supervised or basic model.
    :param activation: The activation function to use in the model.
    :param indiv: Bool indicating whether to run experiment on all patients jointly or separately.
    :param missingness: The percentage of missing data to be used in the experiment.
    :return: MSEs and gMSes for train and test sets
    """
    mses_train = []
    mses_test = []
    CONV_INPUT_LENGTH = 288
    TRAIN_TEST_SPLIT = split
    ADAM_LEARNING_RATE = lr
    ACTIVATION_FUNCTION = activation
    NUM_EPOCHS = epochs
    MINIBATCH_SIZE = batch_size
    print('Beginning experiment...')
    print('Configurations: self_sup: {}, indiv: {} split: {}, lr: {}, epochs: {}, batch_size: {}, missingness: {}'
          .format(self_sup, indiv, split, lr, epochs, batch_size, missingness))
    if missingness < 0.0 or missingness >= 1.0:
        raise ValueError('Missingness must be between 0 and 1.')
    for i in range(0,31):
        # Skip patient 0 if running individual experiments
        if i == 0:
            if indiv:
                continue
        # Read in data
        print(f'Reading in basic data for Patient {i}...')
        CSV_INPUT_BASIC = f'dev/basic_{i}.csv'
        print(f'Reading in self-supervised data for Patient {i}...')
        CSV_INPUT_SELF = f'dev/self_{i}.csv'
        df_basic_raw = pd.read_csv(os.path.join('..', CSV_INPUT_BASIC), )
        df_self_raw = pd.read_csv(os.path.join('..', CSV_INPUT_SELF), )
        df_basic = df_basic_raw.copy(deep=True)
        df_self = df_self_raw.copy(deep=True)
        # Add missingness
        if missingness != 0:
            print('Randomly removing {}% of the data...'.format(missingness*100))
            n = df_basic.shape[0]
            p = 1 - missingness
            true_count = int(n * p)
            chosen_idx = [False] * n

            # Generate random indices for True entries
            true_indices = random.sample(range(n), true_count)

            # Set True entries at the randomly selected indices
            for index in true_indices:
                chosen_idx[index] = True

            df_basic = df_basic.iloc[chosen_idx]

        if dev:
            df_basic = df_basic.iloc[0:100]
            df_self = df_self.iloc[0:100]

        print(f'Creating train-test split for Patient {i}...')
        if i == 0:
            X_train_basic, X_test_basic, Y_train_basic, Y_test_basic = \
                get_train_test_split_all(df_basic, TRAIN_TEST_SPLIT, False)
            X_train_self, X_test_self, Y_train_self, Y_test_self = \
                get_train_test_split_all(df_self, TRAIN_TEST_SPLIT, True)
        else:
            X_train_basic, X_test_basic, Y_train_basic, Y_test_basic = \
                get_train_test_split(df_basic, TRAIN_TEST_SPLIT, False)
            X_train_self, X_test_self, Y_train_self, Y_test_self = \
                get_train_test_split(df_self, TRAIN_TEST_SPLIT, True)
        if self_sup:
            print('Building self-supervised model...')
        else:
            print('Building model...')
        conv_model = get_model(CONV_INPUT_LENGTH, ACTIVATION_FUNCTION, self_sup)
        # Train model
        if self_sup:
            print('Initiating self-supervised training...')
            conv_model = train_model(conv_model, NUM_EPOCHS, X_train_self, X_test_self, Y_train_self, Y_test_self,
                                     ADAM_LEARNING_RATE, MINIBATCH_SIZE)
            print('Getting pretrained model...')
            conv_model = get_finetune_model(conv_model)

        print('Initiating supervised training...')
        conv_model = train_model(conv_model, NUM_EPOCHS, X_train_basic, X_test_basic, Y_train_basic, Y_test_basic,
                                 ADAM_LEARNING_RATE, MINIBATCH_SIZE)

        # Create predictions
        print('Creating predictions for Patient {}...'.format(i))
        preds_train = conv_model.predict(X_train_basic)
        preds_test = conv_model.predict(X_test_basic)
        print('Plotting results for Patient {}...'.format(i))
        # Plot predictions
        plot_predictions(preds_train, Y_train_basic, i, self_sup, False, indiv)
        plot_predictions(preds_test, Y_test_basic, i, self_sup, True, indiv)
        # Calculate MSE
        mses_train.append(mean_squared_error(Y_train_basic, preds_train))
        mses_test.append(mean_squared_error(Y_test_basic, preds_test))
        if i == 0:
            break
    return mses_train, mses_test



if __name__ == '__main__':
    train, test = run_experiment(indiv=True, epochs=10, dev = False, missingness=0.1)
    print(train)
    print(np.mean(train))
    print(test)
    print(np.mean(test))
    #TODO: add normalization (remember to set usebias=false) and regularization
    #TODO: run everything to test
    #TODO: add proper logging/writing to file code
    #TODO: for hyperparameter search, isolate training from df loading
