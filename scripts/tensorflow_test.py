"""TensorFlow low-level implementation test script.

This script tests a low-level implementation of a neural network that does not
use the built-in Keras framework. This is useful because some of the deep RL
formulations require objective functions that differ from those typically used
in supervised learning.

This script generates noise-corrupted samples from a known nonlinear function
and trains a neural network to fit this function.
"""

# Import
import sys
import os
import numpy as np
import functools
import tensorflow as tf

# Custom modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(src_dir)
import neuropy as neu
import neuropy.temp as tmp

print('TensorFlow version: {}'.format(tf.__version__))


# Define functions
def main():
    """Main script function."""
    # Define save directory
    save_dir = os.path.join(
        os.path.expanduser('~'), 'results', 'rl', 'tf_test'
    )
    os.makedirs(save_dir, exist_ok=True)

    # Data parameters
    n_samples = 10000  # Number of samples to generate

    # Model parameters
    n_units = [20, 10]  # Number of units in the hidden layers
    n_layers = len(n_units)

    # Training parameters
    learning_rate = 0.01
    n_epochs = 20000
    n_batches = 5

    # Generate samples
    X_data, Y_data = generate_samples(n_samples)

    # Define variables -- this can certainly be cleaned up in a loop
    v = define_vars(n_units, X_data.shape)
    var_list = [v['W'], v['b']]

    # Define optimizer -- use SGD
    loss_fn = functools.partial(run_model, X_data, Y_data, v)
    opt = tf.optimizers.SGD(learning_rate=learning_rate)

    # Iterate over epochs
    loss_vals = []
    for epoch in range(n_epochs):
        # Take a gradient step, compute loss
        opt.minimize(loss_fn, var_list)
        loss = run_model(X_data, Y_data, v)
        loss_vals.append(loss.numpy())

    # Plot predictions
    y_pred = forward_pass(X_data, v)
    y_pred = y_pred.numpy()
    plot_data(X_data, Y_data, y_pred, save_dir=save_dir, name='Predictions')
    plot_loss(loss_vals, save_dir=save_dir)

    return None


def generate_samples(n):
    """Generate samples from non-linear function."""
    # Sample input features
    X = np.random.default_rng().uniform(0, 2*np.pi, n)  # Input features x samples
    X = np.expand_dims(X, axis=0)
    # Predict output
    Y = f(X)  # Output features x samples

    # Convert to 32-point precision
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    return X, Y


def f(x):
    """Non-linear function"""
    y = np.sin(x)
    y = y + np.random.default_rng().normal(0, 0.1, y.shape)
    return y


def define_vars(n_units, input_shape):
    """Define model parameters/variables."""
    w_init = tf.initializers.glorot_normal()
    b_init = tf.zeros_initializer()
    W_1 = tf.Variable(
        name='W1',
        initial_value=w_init(shape=[n_units[0], input_shape[0]], dtype=tf.float32)
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
    v = {
        'W': [W_1, W_2, W_3],
        'b': [b_1, b_2, b_3]
    }

    return v


@tf.function
def run_model(X, Y, params):
    """Perform one pass of network, including loss calculation"""
    # Forward pass
    y_pred = forward_pass(X, params)
    # Calculate loss
    loss = tf.math.reduce_mean(tf.math.square(Y - y_pred))

    return loss


@tf.function
def forward_pass(X, params):
    """Forward pass of network."""
    W = params['W']
    b = params['b']

    # Run computation -- note that the index starts at zero here
    z_0 = W[0] @ X + b[0]
    a_0 = tf.nn.tanh(z_0)
    z_1 = W[1] @ a_0 + b[1]
    a_1 = tf.nn.tanh(z_1)
    z_2 = W[2] @ a_1 + b[2]
    y_pred = z_2  # Linear activation for output layer

    return y_pred


def plot_data(x, y, y_pred=None, save_dir='', name=''):
    """Plot training data."""
    # Setup figure
    fh, axh = tmp.subplot_fixed(
        1,
        1,
        [300, 300],
        x_margin=[100, 100],
        y_margin=[100, 100]
    )
    # Plot data
    axh[0][0].scatter(x, y, color='k', alpha=0.5, s=1)
    if y_pred is not None:
        axh[0][0].scatter(x, y_pred, color='r', alpha=0.5)
    axh[0][0].set_title(name)

    # Save figure
    fig_name = name + '.pdf'
    fh.savefig(os.path.join(save_dir, fig_name))

    return None


def plot_loss(loss, save_dir='', name='Loss'):
    """Plot loss values."""
    # Setup figure
    fh, axh = tmp.subplot_fixed(
        1,
        1,
        [300, 300],
        x_margin=[100, 100],
        y_margin=[100, 100]
    )
    # Plot data
    axh[0][0].plot(loss, color='k')
    axh[0][0].set_xlabel('Epoch')
    axh[0][0].set_ylabel('Loss')

    # Save figure
    fig_name = name + '.pdf'
    fh.savefig(os.path.join(save_dir, fig_name))

    return None


if __name__ == "__main__":
    main()
