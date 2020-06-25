"""TensorFlow low-level implementation test script.

This script tests a low-level implementation of a neural network that does not
use the built-in Keras framework. This is useful because some of the deep RL
formulations require objective functions that differ from those typically used
in supervised learning.

This script generates noise-corrupted samples from a known nonlinear function
and trains a neural network to fit this function.
"""

# Import
import numpy as np
import tensorflow as tf
print('TensorFlow version: {}'.format(tf.__version__))


# Define functions
def main():
    """Main script function."""
    # Data parameters
    n_samples = 1000  # Number of samples to generate

    # Model parameters
    n_layers = 2  # Number of layers
    n_units = [9, 6]  # Number of units in the hidden layers

    # Training parameters
    learning_rate = 0.01
    n_epochs = 200
    n_batches = 5

    # Generate samples
    X_data, Y_data = generate_samples(n_samples)

    # Define variables -- this can certainly be cleaned up in a loop
    w_init = tf.initializers.glorot_normal()
    b_init = tf.zeros_initializer()
    W_1 = tf.Variable(
        name='W1',
        initial_value=w_init(shape=[n_units[0], X_data.shape[0]], dtype=tf.float32)
    )
    b_1 = tf.Variable(
        name='b1',
        initial_value=b_init(shape=[n_units[0], 1], dtype=tf.float32)
    )
    W_2 = tf.Variable(
        name='W2',
        initial_value=w_init(shape=[n_units[1], n_units[0]], dtype=tf.float32)
    )
    b_2 = tf.Variable(
        name='b2',
        initial_value=b_init(shape=[n_units[1], 1], dtype=tf.float32)
    )
    W_3 = tf.Variable(
        name='W3',
        initial_value=w_init(shape=[1, n_units[1]], dtype=tf.float32)
    )
    b_3 = tf.Variable(
        name='b3',
        initial_value=b_init(shape=[1, 1], dtype=tf.float32)
    )
    params = {
        'W': [W_1, W_2, W_3],
        'b': [b_1, b_2, b_3]
    }

    # Run model -- currently this won't do anything
    loss = run_model(X_data, Y_data, params)

    # TODO: Add outer training loop
    # TODO: look into optimizers and gradients
    return None


def generate_samples(n):
    """Generate samples from non-linear function."""
    # TODO: implement this
    # Sample input features
    X = []  # Input features x samples
    # Predict output
    Y = []  # Output features x samples

    return X, Y


def f(x):
    """Non-linear function"""
    # TODO: implement this
    y = []
    return y


@tf.function
def run_model(X, Y, params):
    """Forward pass of network, including loss calculation"""
    W = params['W']
    b = params['b']

    # Run computation -- note that the index starts at zero here
    z_0 = W[0] @ X + b[0]
    a_0 = tf.nn.relu(z_0)
    z_1 = W[1] @ a_0 + b[1]
    a_1 = tf.nn.relu(z_1)
    z_2 = W[2] @ a_1 + b[2]
    y_pred = z_2  # Linear activation for output layer

    # Define loss function
    loss = tf.math.reduce_mean(tf.math.square(Y - y_pred))

    return loss


if __name__ == "__main__":
    main()
