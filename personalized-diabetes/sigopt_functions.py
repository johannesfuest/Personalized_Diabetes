import sys

sys.path.append("..")
import pandas as pd
import tensorflow.keras.layers as tfl
import tensorflow as tf

pd.set_option("display.max_rows", 1000)
import warnings
import random
import numpy as np

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class GlucoseLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        pass


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, CONV_INPUT_LENGTH: int, run):
        super(ConvLayer, self).__init__()
        self.CONV_INPUT_LENGTH = CONV_INPUT_LENGTH
        self.conv1 = tfl.Conv1D(
            filters=run.params.filter_1,
            kernel_size=run.params.kernel_1,
            strides=run.params.stride_1,
            padding="valid",
            use_bias=False,
        )
        self.norm1 = tfl.BatchNormalization(axis=2)
        self.pool1 = tfl.MaxPool1D(
            pool_size=run.params.pool_size_1,
            strides=run.params.pool_stride_1,
            padding="valid",
        )
        self.drop1 = tfl.Dropout(rate=run.params.dropout_rate)
        self.conv2 = tfl.Conv1D(
            filters=run.params.filter_2,
            kernel_size=run.params.kernel_2,
            strides=run.params.stride_2,
            padding="valid",
            use_bias=False,
        )
        self.norm2 = tfl.BatchNormalization(axis=2)
        self.pool2 = tfl.MaxPool1D(
            pool_size=run.params.pool_size_2,
            strides=run.params.pool_stride_2,
            padding="valid",
        )
        self.drop2 = tfl.Dropout(rate=run.params.dropout_rate)
        self.flatten = tfl.Flatten()

    def build(self, input_shape):
        pass

    def call(self, input):
        assert input.shape[1] == self.CONV_INPUT_LENGTH
        # 1st CONV
        conv1_out = self.conv1(input)
        # Batch Norm
        norm1_out = self.norm1(conv1_out)
        # Max Pool
        pool1_out = self.pool1(norm1_out)
        # Dropout
        drop1_out = self.drop1(pool1_out)
        # 2nd CONV
        conv2_out = self.conv2(drop1_out)
        # Batch Norm
        norm2_out = self.norm2(conv2_out)
        # Max Pool
        pool2_out = self.pool2(norm2_out)
        # Dropout
        drop2_out = self.drop2(pool2_out)
        # Now flatten the Matrix into a 1D vector (shape 1x204)
        flatten_out = self.flatten(drop2_out)
        return flatten_out


class GlucoseModel:
    def get_model(self, CONV_INPUT_LENGTH: int, self_sup: bool):
        # define the input with specified shape
        input = tf.keras.Input(shape=(CONV_INPUT_LENGTH * 4, 1), batch_size=None)
        # Calculate batchsize of current run.
        batch_size = tf.shape(input)[0]
        # Slice the data into four equally sized 1D-chunks for CNN.
        diabetes_input = tf.slice(
            input,
            begin=[0, CONV_INPUT_LENGTH * 0, 0],
            size=[batch_size, CONV_INPUT_LENGTH, 1],
            name="diabetes_input",
        )
        meal_input = tf.slice(
            input,
            begin=[0, CONV_INPUT_LENGTH * 1, 0],
            size=[batch_size, CONV_INPUT_LENGTH, 1],
        )
        smbg_input = tf.slice(
            input,
            begin=[0, CONV_INPUT_LENGTH * 2, 0],
            size=[batch_size, CONV_INPUT_LENGTH, 1],
        )
        excercise_input = tf.slice(
            input,
            begin=[0, CONV_INPUT_LENGTH * 3, 0],
            size=[batch_size, CONV_INPUT_LENGTH, 1],
        )

        # Create the four custom conv-layers
        diabetes_conv = ConvLayer(CONV_INPUT_LENGTH, self.run)(diabetes_input)
        meal_conv = ConvLayer(CONV_INPUT_LENGTH, self.run)(meal_input)
        smbg_conv = ConvLayer(CONV_INPUT_LENGTH, self.run)(smbg_input)
        excercise_conv = ConvLayer(CONV_INPUT_LENGTH, self.run)(excercise_input)

        # Concat the result of conv-layers
        post_conv = tf.concat(
            [diabetes_conv, meal_conv, smbg_conv, excercise_conv],
            axis=1,
            name="post_conv",
        )

        # Now fully connect layers
        # Use multiples of two as recommended in class.
        FC1 = tfl.Dense(units=512, activation=self.run.params.activation)(post_conv)
        DR1 = tfl.Dropout(rate=self.run.params.dropout_rate)(FC1)
        FC2 = tfl.Dense(units=256, activation=self.run.params.activation)(DR1)
        DR2 = tfl.Dropout(rate=self.run.params.dropout_rate)(FC2)
        FC3 = tfl.Dense(units=128, activation=self.run.params.activation)(DR2)
        DR3 = tfl.Dropout(rate=self.run.params.dropout_rate)(FC3)
        FC4 = tfl.Dense(units=64, activation=self.run.params.activation)(DR3)

        # The output does NOT have an activation (regression task)
        # Last layer has 4*CONV_INPUT_LENGTH units if self-supervised, else 1 unit.
        if self_sup:
            output = tfl.Dense(units=4 * CONV_INPUT_LENGTH, activation=None)(FC4)
        else:
            output = tfl.Dense(units=1, activation=None)(FC4)
        model = tf.keras.Model(inputs=input, outputs=output)
        return model

    def __init__(self, CONV_INPUT_LENGTH: int, self_sup: bool, run):
        self.run = run
        self.model = self.get_model(CONV_INPUT_LENGTH, self_sup)

    def train_model(self, epochs, X_train, X_test, Y_train, Y_test, lr, batch_size):
        """
        A function that trains the given model on the given data.
        :param epochs: The number of epochs we want to train for
        :param X_train: The training data predictors
        :param X_test: The testing data predictors
        :param Y_train: The training data labels
        :param Y_test: The testing data labels
        :param lr: The learning rate
        :param batch_size: The batch size
        :return:
        """
        # Create optimizer (Adam with specified learning rate - use default parameters otherwise. )
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # TODO add optional gMSE loss function
        # Compile model - use mse for now.
        self.model.compile(optimizer=adam_optimizer, loss=gMSE, metrics=["mse"])
        # Create train and test datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(
            batch_size
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(
            batch_size
        )
        # Let's run this!
        print("Training model...")
        self.model.fit(
            train_dataset, epochs=epochs, validation_data=test_dataset, verbose=0
        )
        print("Done training model.")

    def evaluate_model(self, X_test, Y_test):
        """
        A function that evaluates the model on the given data.
        :param model: The model we want to evaluate
        :param X_test: The testing data predictors
        :param Y_test: The testing data labels
        :return: The loss and mse of the model on the given data
        """
        # Evaluate model
        metrics = self.model.evaluate(X_test, Y_test, verbose=0)
        return metrics

    def activate_finetune_mode(self):
        # Create new output layer which uses output of before-last layer as input
        output = tfl.Dense(units=1, activation=None)(self.model.layers[-2].output)
        # Create new model
        finetune_model = tf.keras.Model(inputs=self.model.input, outputs=output)
        self.model = finetune_model


def get_train_test_split(df, TRAIN_TEST_SPLIT: float, self_sup: bool):
    """
    A function that takes a dataframe and returns a temporal train and test split using the given argument to determine
    split.
    :param df: the dataframe to be split
    :param TRAIN_TEST_SPLIT: the split to be used as a number between 0 and 1
    :param self_sup: Boolean indicating whether dataframe is self-supervised or not
    :return: X_train, X_test, Y_train, Y_test according to the given split
    """
    df = df.sort_values("LocalDtTm")
    # split into train and test
    train_length = int(TRAIN_TEST_SPLIT * df.shape[0])
    train = df.iloc[:train_length, :]
    test = df.iloc[train_length:, :]
    assert (
        test.shape[0] + train.shape[0] == df.shape[0]
    ), "Train-Test shapes don not add up."
    if self_sup:
        X_train = train.drop(columns=["LocalDtTm", "CGM"])
        Y_train = train.drop(columns=["LocalDtTm", "CGM"])
        X_test = test.drop(columns=["LocalDtTm", "CGM"])
        Y_test = test.drop(columns=["LocalDtTm", "CGM"])
        for i in range(1, 289):
            X_train = X_train.drop(
                columns=[
                    f"insulin {i} target",
                    f"mealsize {i} target",
                    f"carbs {i} target",
                    f"exercise {i} target",
                ]
            )
            Y_train = Y_train.drop(
                columns=[f"insulin {i}", f"mealsize {i}", f"carbs {i}", f"exercise {i}"]
            )
            X_test = X_test.drop(
                columns=[
                    f"insulin {i} target",
                    f"mealsize {i} target",
                    f"carbs {i} target",
                    f"exercise {i} target",
                ]
            )
            Y_test = Y_test.drop(
                columns=[f"insulin {i}", f"mealsize {i}", f"carbs {i}", f"exercise {i}"]
            )
    else:
        X_train = train.drop(columns=["LocalDtTm", "CGM"])
        Y_train = train[["CGM", "DeidentID"]]
        X_test = test.drop(columns=["LocalDtTm", "CGM"])
        Y_test = test[["CGM", "DeidentID"]]
    return X_train, X_test, Y_train, Y_test


def get_train_test_split_all(df, TRAIN_TEST_SPLIT: float, self_sup: bool):
    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()
    for i in range(1, 31):
        X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = get_train_test_split(
            df[df["DeidentID"] == i], TRAIN_TEST_SPLIT, self_sup
        )
        X_train = pd.concat([X_train, X_train_temp])
        Y_train = pd.concat([Y_train, Y_train_temp])
        X_test = pd.concat([X_test, X_test_temp])
        Y_test = pd.concat([Y_test, Y_test_temp])
    # X_train.drop(columns=['DeidentID'], inplace=True)
    # X_test.drop(columns=['DeidentID'], inplace=True)
    return X_train, X_test, Y_train, Y_test


def get_train_test_split_search(df, TRAIN_TEST_SPLIT: float, self_sup: bool):
    # keep only the first TRAIN_TEST_SPLIT * 100 rows of df
    n = int(TRAIN_TEST_SPLIT * 100 * len(df))
    df = df.head(n)
    return get_train_test_split_all(df, TRAIN_TEST_SPLIT, self_sup)


def apply_data_missingness(df, missingness: float):
    if missingness == 0.0:
        return df
    n = df.shape[0]
    p = 1 - missingness
    true_count = int(n * p)
    chosen_idx = [False] * n

    # Generate random indices for True entries
    true_indices = random.sample(range(n), true_count)

    # Set True entries at the randomly selected indices
    for index in true_indices:
        chosen_idx[index] = True

    df = df.iloc[chosen_idx]
    return df


def xi(x, a, epsilon):
    two = tf.constant(2, dtype=tf.float32)
    two_over_epsilon = tf.math.divide(two, epsilon)
    a_plus_epsilon_over_two = tf.math.add(a, tf.math.divide(epsilon, two))
    return tf.multiply(two_over_epsilon, tf.math.subtract(x, a_plus_epsilon_over_two))
    # return 2/epsilon * (x-a-epsilon/2)


def sigmoid(x, a, epsilon):
    XI = xi(x, a, epsilon)
    zero = tf.constant(0.0, dtype=tf.float32)
    half = tf.constant(0.5, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    two = tf.constant(2, dtype=tf.float32)
    epsilon_over_two = tf.math.divide(epsilon, two)
    three = tf.constant(3, dtype=tf.float32)
    calc = tf.math.add(tf.math.add(tf.math.negative(tf.math.pow(XI, three)), XI), half)
    four = tf.constant(4, dtype=tf.float32)

    term1 = tf.math.add(
        tf.math.multiply(tf.math.negative(half), tf.math.pow(XI, four)), calc
    )
    term2 = tf.math.add(tf.math.multiply(half, tf.math.pow(XI, four)), calc)

    return tf.where(
        tf.less_equal(x, a),
        zero,
        tf.where(
            tf.less_equal(x, tf.math.add(a, epsilon_over_two)),
            term1,
            tf.where(tf.less_equal(x, tf.math.add(a, epsilon)), term2, one),
        ),
    )


def xi_bar(x, a, epsilon):
    minus_two = tf.constant(-2.0, dtype=tf.float32)
    m_two_over_epsilon = tf.math.divide(minus_two, epsilon)
    a_plus_epsilon_over_m_two = tf.math.add(a, tf.math.divide(epsilon, minus_two))
    return tf.multiply(
        m_two_over_epsilon, tf.math.subtract(x, a_plus_epsilon_over_m_two)
    )
    # return -2/epsilon * (x-a+epsilon/2)


def sigmoid_bar(x, a, epsilon):
    XI_BAR = xi_bar(x, a, epsilon)
    zero = tf.constant(0.0, dtype=tf.float32)
    half = tf.constant(0.5, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    two = tf.constant(2, dtype=tf.float32)
    three = tf.constant(3, dtype=tf.float32)
    four = tf.constant(4, dtype=tf.float32)
    epsilon_over_two = tf.math.divide(epsilon, two)
    calc = tf.math.add(
        tf.math.add(tf.math.negative(tf.math.pow(XI_BAR, three)), XI_BAR), half
    )
    term1 = tf.math.add(
        tf.math.multiply(tf.math.negative(half), tf.math.pow(XI_BAR, four)), calc
    )
    term2 = tf.math.add(tf.math.multiply(half, tf.math.pow(XI_BAR, four)), calc)

    # print(XI)
    return tf.where(
        tf.less_equal(x, tf.math.subtract(a, epsilon)),
        one,
        tf.where(
            tf.math.less_equal(x, tf.math.subtract(a, epsilon_over_two)),
            term2,
            tf.where(tf.math.less_equal(x, a), term1, zero),
        ),
    )


alpha_L = tf.constant(1.5, dtype=tf.float32)
alpha_H = tf.constant(1.0, dtype=tf.float32)
beta_L = tf.constant(30.0, dtype=tf.float32)
beta_H = tf.constant(100.0, dtype=tf.float32)
gamma_L = tf.constant(10.0, dtype=tf.float32)
gamma_H = tf.constant(20.0, dtype=tf.float32)
t_L = tf.constant(85.0, dtype=tf.float32)
t_H = tf.constant(155.0, dtype=tf.float32)


def Pen(
    g,
    g_hat,
    alpha_L=alpha_L,
    alpha_H=alpha_H,
    beta_L=beta_L,
    beta_H=beta_H,
    gamma_L=gamma_L,
    gamma_H=gamma_H,
    t_L=t_L,
    t_H=t_H,
):
    one = tf.constant(1.0, dtype=tf.float32)
    return tf.math.add(
        one,
        tf.math.add(
            tf.math.multiply(
                alpha_L,
                tf.math.multiply(
                    sigmoid_bar(g, t_L, beta_L), sigmoid(g_hat, g, gamma_L)
                ),
            ),
            tf.math.multiply(
                alpha_H,
                tf.math.multiply(
                    sigmoid(g, t_H, beta_H), sigmoid_bar(g_hat, g, gamma_H)
                ),
            ),
        ),
    )


def gSE(
    g,
    g_hat,
    alpha_L=alpha_L,
    alpha_H=alpha_H,
    beta_L=beta_L,
    beta_H=beta_H,
    gamma_L=gamma_L,
    gamma_H=gamma_H,
    t_L=t_L,
    t_H=t_H,
):
    return tf.math.multiply(
        tf.math.square(tf.subtract(tf.cast(g, tf.float32), g_hat)),
        Pen(
            tf.cast(g, tf.float32),
            g_hat,
            alpha_L,
            alpha_H,
            beta_L,
            beta_H,
            gamma_L,
            gamma_H,
            t_L,
            t_H,
        ),
    )


def gMSE(
    g,
    g_hat,
    alpha_L=alpha_L,
    alpha_H=alpha_H,
    beta_L=beta_L,
    beta_H=beta_H,
    gamma_L=gamma_L,
    gamma_H=gamma_H,
    t_L=t_L,
    t_H=t_H,
):
    return tf.math.reduce_mean(
        gSE(g, g_hat, alpha_L, alpha_H, beta_L, beta_H, gamma_L, gamma_H, t_L, t_H)
    )


if __name__ == "__main__":
    print("stop being red")
    # TODO: ensure dfs are sorted properly
    # TODO: discuss batch norm in FC layers -> ask Peter
    # TODO: Go over data visualization Functions (check naming convention and titles, etc.)
    # TODO: Think about modelling justification
    # TODO: Merge all to main
    # TODO: Run Grid Search for each Model
    # TODO: Run Experiments for each Model
