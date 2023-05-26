
import sys
sys.path.append('..')
import os

import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
import tensorflow as tf
#import mean squared error
from sklearn.metrics import mean_squared_error

import datetime
pd.set_option('display.max_rows', 1000)
# Future TODO: Create custom loss.
class GlucoseLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        pass

def hyper_metric(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1,))
    y_pred = tf.reshape(y_pred, shape=(-1,))
    preds_while_hyper = tf.boolean_mask(y_pred, mask=tf.greater(y_true, 100.0))
    return tf.reduce_mean(tf.cast(tf.greater(preds_while_hyper, 100.0), tf.float32))

# Custom metric: Share of correctly recognized hypo glucose states.
def hypo_metric(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1,))
    y_pred = tf.reshape(y_pred, shape=(-1,))
    preds_while_hyper = tf.boolean_mask(y_pred, mask=tf.less(y_true, 70.0))
    return tf.reduce_mean(tf.cast(tf.less(preds_while_hyper, 70.0), tf.float32))


# In order to avoid code duplication, create a custom "ConvLayer" that will be created four times.
# See Figure 1 in milestone report.
class ConvLayer(tf.keras.layers.Layer):
    # constructor method, initializing the object's attributes
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = tfl.Conv1D(filters=3, kernel_size=5, strides=1, padding='valid')
        # 1st Normalization
        self.norm1 = tfl.BatchNormalization(axis = 2) # axis = 2 refers to the dimension of the channels, so norm. along the channels
        self.pool1 = tfl.MaxPool1D(pool_size=2, strides=2, padding='valid')
        # add 1st dropout
        self.drop1 = tfl.Dropout(rate = 0.25)
        self.conv2 = tfl.Conv1D(filters=6, kernel_size=5, strides=1, padding='valid')
        # 2nd Normalization
        self.norm2 = tfl.BatchNormalization(axis = 2)
        self.pool2 = tfl.MaxPool1D(pool_size=6, strides=4, padding='valid')
        # add 2nd dropout
        self.drop2 = tfl.Dropout(rate = 0.25)
        self.flatten = tfl.Flatten()

    def build(self, input_shape):
        pass

    # invoked every time the layer is called
    def call(self, input):
        assert input.shape[1] == CONV_INPUT_LENGTH
        # 1st CONV
        conv1_out = self.conv1(input)
        # 1st NORM
        norm1_out = self.norm1(conv1_out)
        # Max Pool
        pool1_out = self.pool1(norm1_out)
        # add 1st dropout
        drop1_out = self.drop1(pool1_out)
        # 2nd CONV
        conv2_out = self.conv2(drop1_out)
        # 2nd NORM
        norm2_out = self.norm2(conv2_out)
        # Max Pool
        pool2_out = self.pool2(norm2_out)
        # add 2nd dropout
        drop2_out = self.drop2(pool2_out)
        # add 2nd dropout 
        
        # Now flatten the Matrix into a 1D vector (shape 1x204)
        flatten_out = self.flatten(drop2_out)

        return flatten_out


def get_model():
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
    diabetes_conv = ConvLayer()(diabetes_input)
    meal_conv = ConvLayer()(meal_input)
    smbg_conv = ConvLayer()(smbg_input)
    excercise_conv = ConvLayer()(excercise_input)

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
    output = tfl.Dense(units=1, activation=None)(FC4)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model
mses_train = []
mses_test = []

for i in range(1,31):
    CSV_INPUT = f'dev/df_final_{i}.csv'
    CONV_INPUT_LENGTH = 288
    TRAIN_TEST_SPLIT = 0.8
    ADAM_LEARNING_RATE = 0.0001
    ACTIVATION_FUNCTION = 'relu'
    NUM_EPOCHS = 20
    MINIBATCH_SIZE = 32
    # Persist an unaltered deep copy of the data for faster iterations.
    df_raw = pd.read_csv(os.path.join('..', CSV_INPUT), )
    df = df_raw.copy(deep=True)
    # Sort DF for time-based train-test split
    df = df.sort_values('LocalDtTm')
    train_length = int(TRAIN_TEST_SPLIT * df.shape[0])
    df_train = df.iloc[:train_length, :]
    df_test = df.iloc[train_length:, :]
    assert df_test.shape[0] + df_train.shape[0] == df.shape[0], 'Train-Test shapes don not add up.'

    X_train = df_train.drop(columns=['LocalDtTm', 'CGM'])
    Y_train = df_train[['CGM']]

    X_test = df_test.drop(columns=['LocalDtTm', 'CGM'])
    Y_test = df_test[['CGM']]
    conv_model = get_model()
    # Create optimizer (Adam with specified learning rate - use default parameters otherwise. )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=ADAM_LEARNING_RATE)
    # Compile model - use mse for now.
    conv_model.compile(optimizer=adam_optimizer,
                       loss='mse',
                       metrics=['mse', hypo_metric, hyper_metric])
    conv_model.summary()
    print(conv_model.summary())
    # Create train and test datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(MINIBATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(MINIBATCH_SIZE)
    # Let's run this!
    history = conv_model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)
    # Create predictions
    preds_train = conv_model.predict(X_train)
    preds_test = conv_model.predict(X_test)
    # plot train predictions against actual train values
    plt.scatter(Y_train, preds_train)
    plt.xlabel('True Values [Glucose]')
    plt.ylabel('Predictions [Glucose]')
    plt.axis('equal')
    plt.axis('square')
    plt.title(f'train_{i}')
    #save plot
    plt.savefig(f'train_{i}.png')
    plt.show()
    mses_train.append(mean_squared_error(Y_train, preds_train))
    # plot test predictions against actual test values
    plt.scatter(Y_test, preds_test)
    plt.xlabel('True Values [Glucose]')
    plt.ylabel('Predictions [Glucose]')
    plt.axis('equal')
    plt.axis('square')
    plt.title(f'test_{i}')
    #save plot
    plt.savefig(f'test_{i}.png')
    plt.show()
    #print mse
    mses_test.append(mean_squared_error(Y_test, preds_test))
    print(f'MSE {i} : {mean_squared_error(Y_test, preds_test)}')

print(mses_test)
print(np.mean(mses_test))