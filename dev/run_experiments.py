import tensorflow as tf
import sigopt
import os
import logging
import PIL
import numpy as np
import matplotlib.pyplot as plt
import io
from plotly.io import to_image
import plotly.express as px
import git



os.environ["SIGOPT_API_TOKEN"] = "CDLCFJJUWDYYKMDCXOISTWNALSSWLQQGBJHEBNVKXFQMFWNE"
os.environ["SIGOPT_PROJECT"] = "personalized-diabetes"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filename='log.log')



def load_data():
    return tf.keras.datasets.mnist.load_data()

class KerasNNModel:
    def __init__(self, hidden_layer_size, activation_fn):
        model = tf.keras.Sequential(
        [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_layer_size, activation=activation_fn),
        tf.keras.layers.Dense(10),
        ]
        )
        self.model = model

    def get_keras_nn_model(self):
        return self.model

    def train_model(self, train_images, train_labels, optimizer_type, metrics_list, num_epochs):
        self.model.compile(
            optimizer=optimizer_type,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=metrics_list,
            )
        self.model.fit(train_images, train_labels, epochs=num_epochs)

    def evaluate_model(self, test_images, test_labels):
        metrics_dict = self.model.evaluate(test_images, test_labels, verbose=2, return_dict=True)

        # Log image
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        image_bytes = to_image(fig, format='png')
        run.log_image(PIL.Image.open(io.BytesIO(image_bytes)), name='lol')
        return metrics_dict


def load_data_train_model(run, data):
    run.log_dataset(name="mnist")
    (train_images, train_labels), (test_images, test_labels) = data

  # set model training, architecture parameters and hyperparameters
    run.params.num_epochs = 2

    # create the model
    keras_nn_model = KerasNNModel(
    hidden_layer_size=run.params.hidden_layer_size, activation_fn=run.params.activation_function
    )
    run.log_model("Keras NN Model with 1 Hidden layer")

    # train the model
    keras_nn_model.train_model(train_images, train_labels, "adam", ["accuracy"], run.params.num_epochs)
    run.log_metadata("sgd optimizer", "adam")
    metrics_dict = keras_nn_model.evaluate_model(test_images, test_labels)

    # log performance metrics
    run.log_metric("holdout_accuracy", metrics_dict["accuracy"])

if __name__ == "__main__":
    data = load_data()
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    experiment = sigopt.create_experiment(
        name="Marcos_Test_Experiment",
        type="offline",
        parameters=[
        dict(name="hidden_layer_size", type="int", bounds=dict(min=32, max=512)),
        dict(name="activation_function", type="categorical", categorical_values=["relu", "tanh"]),
        ],
        metrics=[dict(name="holdout_accuracy", strategy="optimize", objective="maximize")],
        parallel_bandwidth=1,
        budget=2,
        )
    for run in experiment.loop():
        with run:
          run.log_metadata('commit', sha)
          load_data_train_model(run=run, data=data)
