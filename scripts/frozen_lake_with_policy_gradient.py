"""Implementation of Policy Gradient learning using the Frozen Lake environment.

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
    n_units = [20, 20, 10, 5]  # Number of units in the hidden layers
    n_layers = len(n_units)

    # Training parameters
    learning_rate = 0.01
    n_epochs = 20000
    n_batches = 5

    # Generate samples
    X_data, Y_data = generate_samples(n_samples)

    # Initialize policy network
    network = PolicyNetwork(n_units)
    _ = network(X_data)  # Initialize the network by passing an input
    var_list = network.trainable_weights

    # Define optimizer -- use SGD
    opt = tf.optimizers.SGD(learning_rate=learning_rate)

    # The general algorithm:
    # Initialize policy network based on the environment:
    #   - n_inputs: number of states
    #   - n_outputs: action space size ((None, 4) for frozen lake)
    #
    # Initialize policy network
    # Training loop:
    #   - Roll out policy (collect many episodes)
    #   - Calculate gradients, update policy network

    # TODO: Update policy network to output probabilities
    # TODO: add softmax layer at end, sample to select action

    # Functions needed:
    # - Action selection (given a network)
    # - Policy rollout (generate training episodes)
    # - New loss function (adding rewards)
    # - Wrapper around policy update (just take a single gradient step)
    # - Update gradient step to maximize rewards (rather than minimimze loss)?
    #
    # Note -- might want to create a Policy class, which can have methods such
    # as sample_action(), update(), etc.
    #
    # We will also need the following:
    # - A way to save the network weights (& re-instantiate)
    # - Possibly consider ways to parallelize
    # - Tracking of rewards.  We will want to monitor this to make sure that
    #   the number of rewards is increasing over time.

    # Iterate over epochs
    loss_vals = []
    for epoch in range(n_epochs):
        # Compute gradients
        with tf.GradientTape() as tape:
            # Evaluate loss
            y_pred = network(X_data)
            loss = tf.math.reduce_mean(tf.math.square(Y_data - y_pred))

        # Get gradients
        grads = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(grads, var_list))
        loss_vals.append(loss.numpy())

        # Display update
        if epoch % 100 == 0:
            # Display update
            print('Epoch {}: Loss = {}'.format(epoch, loss.numpy()))

    # Plot predictions
    y_pred = network(X_data)
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

    # Convert to 32-point precision and transpose (TF assumes the first
    # dimension corresponds to batches.
    X = X.astype(np.float32).T
    Y = Y.astype(np.float32).T

    return X, Y


def f(x):
    """Non-linear function"""
    y = np.sin(x)
    y = y + np.random.default_rng().normal(0, 0.1, y.shape)
    return y


class PolicyNetwork(tf.keras.layers.Layer):
    """Simple policy multi-layer policy network."""
    def __init__(self, n_units):
        super(PolicyNetwork, self).__init__()
        # Create hidden layers
        self.layers = [
            tf.keras.layers.Dense(
                n,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                bias_initializer=tf.keras.initializers.Constant()
            )
            for n in n_units
        ]
        # Add output layer
        self.layers.append(
            tf.keras.layers.Dense(1, activation=None)
        )

    def call(self, inputs, **kwargs):
        """Compute the output of the network."""
        # Get output from the first layer
        x = self.layers[0](inputs)
        # Iterate over the rest of the layers
        for layer in self.layers[1:]:
            x = layer(x)
        # Return results
        return x


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
