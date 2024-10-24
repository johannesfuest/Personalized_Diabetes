import pandas as pd
import sys
import tensorflow as tf
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import warnings


pd.set_option("display.max_rows", 1000)
sys.path.append("..")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class ConvLayer(tf.keras.layers.Layer):
    """
    Custom subclass on tf.keras.layers.Layer to implement a convolutional layer specific to our problem. Each layer
    contains two 1D Convolutions with max pooling, dropout and batchnorm between.
    Each layer is further defined by the following parameters:
    - CONV_INPUT_LENGTH: the length of the input to the convolutional layer (288 in our case due to 5 minute intervals)
    - fixed_hyperparemeters: a dictionary containing the hyperparameters for the model
    """

    def __init__(self, CONV_INPUT_LENGTH: int, fixed_hyperparameters, **kwargs):
        """Constructor for the ConvLayer class."""
        super(ConvLayer, self).__init__(**kwargs)
        self.CONV_INPUT_LENGTH = CONV_INPUT_LENGTH
        # 1st CONV
        self.conv1 = tfl.Conv1D(
            filters=fixed_hyperparameters["filter_1"],
            kernel_size=fixed_hyperparameters["kernel_1"],
            strides=fixed_hyperparameters["stride_1"],
            padding="valid",
            use_bias=False,
            activation="relu",
        )
        # Batch Norm
        self.norm1 = tfl.BatchNormalization(axis=2)
        # Max Pool
        self.pool1 = tfl.MaxPool1D(
            pool_size=fixed_hyperparameters["pool_size_1"],
            strides=fixed_hyperparameters["pool_stride_1"],
            padding="valid",
        )
        self.drop1 = tfl.Dropout(rate=fixed_hyperparameters["dropout_rate"])
        # 2nd CONV
        self.conv2 = tfl.Conv1D(
            filters=fixed_hyperparameters["filter_2"],
            kernel_size=fixed_hyperparameters["kernel_2"],
            strides=fixed_hyperparameters["stride_2"],
            padding="valid",
            use_bias=False,
            activation="relu",
        )
        # Batch Norm
        self.norm2 = tfl.BatchNormalization(axis=2)
        # Max Pool
        self.pool2 = tfl.MaxPool1D(
            pool_size=fixed_hyperparameters["pool_size_2"],
            strides=fixed_hyperparameters["pool_stride_2"],
            padding="valid",
        )
        # Dropout
        self.drop2 = tfl.Dropout(rate=fixed_hyperparameters["dropout_rate"])
        # Now flatten the Matrix into a 1D vector (shape 1x204)
        self.flatten = tfl.Flatten()

    def build(self, input_shape):
        pass

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "CONV_INPUT_LENGTH": self.CONV_INPUT_LENGTH,
                "conv1": self.conv1,
                "norm1": self.norm1,
                "pool1": self.pool1,
                "drop1": self.drop1,
                "conv2": self.conv2,
                "norm2": self.norm2,
                "pool2": self.pool2,
                "drop2": self.drop2,
                "flatten": self.flatten,
            }
        )
        return config

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
    """
    Class to define the model architecture for the glucose prediction model.
    """

    def get_model(self, CONV_INPUT_LENGTH: int, self_sup: bool):
        """
        Function to define the model architecture for the glucose prediction model.
        :param CONV_INPUT_LENGTH:
        :param self_sup: whether the model should be trained in a self-supervised manner (changes only last layer)
        :return: a tf.keras.Model object containing the model architecture
        """
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
        diabetes_conv = ConvLayer(CONV_INPUT_LENGTH, self.fixed_hyperparameters)(diabetes_input)
        meal_conv = ConvLayer(CONV_INPUT_LENGTH, self.fixed_hyperparameters)(meal_input)
        smbg_conv = ConvLayer(CONV_INPUT_LENGTH, self.fixed_hyperparameters)(smbg_input)
        excercise_conv = ConvLayer(CONV_INPUT_LENGTH, self.fixed_hyperparameters)(excercise_input)

        # Concat the result of conv-layers
        post_conv = tf.concat(
            [diabetes_conv, meal_conv, smbg_conv, excercise_conv],
            axis=1,
            name="post_conv",
        )

        # Now fully connect layers
        # Use multiples of two as recommended in class.
        FC1 = tfl.Dense(units=512, activation="relu")(post_conv)
        DR1 = tfl.Dropout(rate=self.fixed_hyperparameters["dropout_rate"])(FC1)
        FC2 = tfl.Dense(units=256, activation="relu")(DR1)
        DR2 = tfl.Dropout(rate=self.fixed_hyperparameters["dropout_rate"])(FC2)
        FC3 = tfl.Dense(units=128, activation="relu")(DR2)
        DR3 = tfl.Dropout(rate=self.fixed_hyperparameters["dropout_rate"])(FC3)
        FC4 = tfl.Dense(units=64, activation="relu")(DR3)

        # The output does NOT have an activation (regression task)
        # Last layer has 4*CONV_INPUT_LENGTH units if self-supervised, else 1 unit.
        if self_sup:
            output = tfl.Dense(units=4, activation=None)(FC4)
        else:
            FC5 = tfl.Dense(units=1, activation="relu")(FC4)
            output = tfl.ReLU(max_value=401)(FC5)
        model = tf.keras.Model(inputs=input, outputs=output)
        return model

    def __init__(self, CONV_INPUT_LENGTH: int, self_sup: bool, fixed_hyperparameters):
        self.fixed_hyperparameters = fixed_hyperparameters
        self.model = self.get_model(CONV_INPUT_LENGTH, self_sup)

    def set_model(self, model):
        self.model = model

    def train_model(
        self, epochs, X_train, X_test, Y_train, Y_test, lr, batch_size, self_sup: bool, missingness_modulo: int, name: str
    ):
        """
        A function that trains the given model on the given data.
        :param epochs: The number of epochs we want to train for
        :param X_train: The training data predictors
        :param X_test: The testing data predictors
        :param Y_train: The training data labels
        :param Y_test: The testing data labels
        :param lr: The learning rate
        :param batch_size: The batch size
        :param self_sup: Boolean indicating whether dataframe is self-supervised or not
        :param missingness_modulo: The modulo for missingness
        :param name: The name of the model being trained
        """
        # Create optimizer (Adam with specified learning rate - use default parameters otherwise. )
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # Compile model
        if self_sup:
            self.model.compile(optimizer=adam_optimizer, loss="mse", metrics=["mse"])
        else:
            self.model.compile(optimizer=adam_optimizer, loss=gMSE, metrics=["mse"])
        # Create train and test datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(
            batch_size
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(
            batch_size
        )
        validation_steps = len(X_test) // batch_size

        history = self.model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, validation_steps=validation_steps)
        
        # Plot loss evolution
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title(f'Loss Evolution for {name} on every {missingness_modulo}th row')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'loss_evolution_{name}_{missingness_modulo}.png')
        plt.close()
        

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
        """
        A function that activates finetune mode for the model. This means that the last layer is removed and a new one
        that predicts a single value instead of a whole row of predictors is added.
        :return: None, only changes the model
        """
        # Remove last layer and add new one
        tflast = tfl.Dense(units=1, activation=None)(self.model.layers[-2].output)
        output = tfl.ReLU(max_value=401)(tflast)
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
    # Sanity check: train and test should add up to the original dataframe
    assert (
        test.shape[0] + train.shape[0] == df.shape[0]
    ), "Train-Test shapes don not add up."
    if self_sup:
        # Drop the columns that are not needed for self-supervised learning
        X_train = train.drop(
            columns=[
                "LocalDtTm",
                "CGM",
                "future_insulin",
                "future_meal",
                "future_carbs",
                "future_exercise",
            ]
        )
        Y_train = train.drop(columns=["LocalDtTm", "CGM"])
        X_test = test.drop(
            columns=[
                "LocalDtTm",
                "CGM",
                "future_insulin",
                "future_meal",
                "future_carbs",
                "future_exercise",
            ]
        )
        Y_test = test.drop(columns=["LocalDtTm", "CGM"])
        for i in range(1, 289):
            Y_train = Y_train.drop(
                columns=[f"insulin {i}", f"mealsize {i}", f"carbs {i}", f"exercise {i}"]
            )
            Y_test = Y_test.drop(
                columns=[f"insulin {i}", f"mealsize {i}", f"carbs {i}", f"exercise {i}"]
            )
    else:
        # Drop the columns that are not needed for supervised learning in train set
        X_train = train.drop(columns=["LocalDtTm", "CGM"])
        X_test = test.drop(columns=["LocalDtTm", "CGM"])
        # Drop the columns that are not needed for supervised learning in test set
        Y_train = train[["CGM", "DeidentID"]]
        Y_test = test[["CGM", "DeidentID"]]
    return X_train, X_test, Y_train, Y_test


def get_train_test_split_all(df, TRAIN_TEST_SPLIT: float, self_sup: bool):
    """
    A function that takes a dataframe and returns a temporal train and test split using the given argument to determine
    :param df: The dataframe to be split
    :param TRAIN_TEST_SPLIT: The split to be used as a number between 0 and 1
    :param self_sup: Whether the dataframe is self-supervised or not
    :return: X_train, X_test, Y_train, Y_test according to the given split
    """
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
    """
    A function that takes a dataframe and returns a temporal train and test split using the given argument to determine.
    To avoid data snooping in the grid search, this function first removes the testing portion of the data and then
    performs the train-test split.
    :param df: The dataframe to be split
    :param TRAIN_TEST_SPLIT: The split to be used as a number between 0 and 1
    :param self_sup: Whether the dataframe is self-supervised or not
    :return: X_train, X_test, Y_train, Y_test according to the given split
    """
    # keep only the first TRAIN_TEST_SPLIT * 100 rows of df
    n = int(TRAIN_TEST_SPLIT * len(df))
    df = df.head(n)
    return get_train_test_split_all(df, TRAIN_TEST_SPLIT, self_sup)


def apply_data_missingness(x_train, y_train, missingness_modulo: int):
    """
    A function that applies missingness to the data according to the given modulo. Data missingness is achieved by
    removing every nth row from the data to arrive at an evenly spaced missingness pattern. This is done to reflect
    the likely use case for our model (i.e. one CGM measurement every week instead of every 5 minutes).
    :param x_train: The input data
    :param y_train: The target data
    :param missingness_modulo: How many rows to skip
    :return: x_train, y_train with missingness applied
    """
    # Sanity check: x_train and y_train should have the same number of rows
    assert (
        x_train.shape[0] == y_train.shape[0]
    ), "x_train and y_train should have the same number of rows before missingness is applied."
    x_train = x_train[::missingness_modulo]
    y_train = y_train[::missingness_modulo]
    assert (
        x_train.shape[0] == y_train.shape[0]
    ), "x_train and y_train should have the same number of rows after missingness is applied."
    return x_train, y_train


def xi(x, a, epsilon):
    """
    xi function from gMSE paper: 2/epsilon * (x-a-epsilon/2)
    Only use TensorFlow operations to ensure that the gradient is computed correctly and efficiently
    """
    two = tf.constant(2, dtype=tf.float32)
    two_over_epsilon = tf.math.divide(two, epsilon)
    a_plus_epsilon_over_two = tf.math.add(a, tf.math.divide(epsilon, two))
    return tf.multiply(two_over_epsilon, tf.math.subtract(x, a_plus_epsilon_over_two))
    # return 2/epsilon * (x-a-epsilon/2)


def sigmoid(x, a, epsilon):
    """
    sigmoid function from gMSE paper:
    """
    XI = xi(x, a, epsilon)
    # define constants to be used in the function
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
    # case distinction for the sigmoid function
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
    """
    xi_bar function from gMSE paper: -2/epsilon * (x-a+epsilon/2)"""
    minus_two = tf.constant(-2.0, dtype=tf.float32)
    m_two_over_epsilon = tf.math.divide(minus_two, epsilon)
    a_plus_epsilon_over_m_two = tf.math.add(a, tf.math.divide(epsilon, minus_two))
    return tf.multiply(
        m_two_over_epsilon, tf.math.subtract(x, a_plus_epsilon_over_m_two)
    )
    # return -2/epsilon * (x-a+epsilon/2)


def sigmoid_bar(x, a, epsilon):
    """
    sigmoid_bar function from gMSE paper"""
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

    # case distinction for the sigmoid_bar function
    return tf.where(
        tf.less_equal(x, tf.math.subtract(a, epsilon)),
        one,
        tf.where(
            tf.math.less_equal(x, tf.math.subtract(a, epsilon_over_two)),
            term2,
            tf.where(tf.math.less_equal(x, a), term1, zero),
        ),
    )


# Define constants from original gMSE paper
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
    """Penalty function from gMSE paper"""
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
    """gMSE function from gMSE paper: (g-g_hat)^2 * Pen(g, g_hat) = MSE * Pen(g, g_hat)"""
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
    """Mean aggregated gMSE function from gMSE paper: mean(gMSE) = mean(MSE * Pen(g, g_hat))"""
    return tf.math.reduce_mean(
        gSE(g, g_hat, alpha_L, alpha_H, beta_L, beta_H, gamma_L, gamma_H, t_L, t_H)
    )


if __name__ == "__main__":
    sys.exit(0)
