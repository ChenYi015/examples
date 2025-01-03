#!/usr/bin/env python

import argparse
import datetime
import gzip
import os
import time

import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="",
        metavar="D",
        help="directory of MNIST dataset (default: '')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        '--dir',
        default=os.path.join(os.path.dirname(__file__), 'logs'),
        metavar='L',
        help='directory where summary logs are stored'
    )
    args = parser.parse_args()
    print(f'args: {args}')
    return args


def create_model(args):
    """Create a simple model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(28, 28)),
        tf.keras.layers.Reshape((28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


def load_data(args, buffer_size, batch_size):
    """Load MNIST dataset."""
    if args.data_dir:
        # Load MNIST dataset from local directory
        print(f'Loading MNIST dataset from local directory: {args.data_dir}.')
        with gzip.open(os.path.join(args.data_dir, 'train-images-idx3-ubyte.gz'), 'rb') as f:
            x_train = np.frombuffer(
                f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(os.path.join(args.data_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
            y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        with gzip.open(os.path.join(args.data_dir, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
            x_test = np.frombuffer(
                f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(os.path.join(args.data_dir, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
            y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    else:
        # Download MNIST dataset from keras
        print(f'Loading MNIST dataset from keras.')
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    ds_train = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)

    return ds_train, ds_test


def get_callbacks(args):
    """Get callbacks."""
    callbacks = []

    if args.dir:
        log_dir = os.path.join(
            args.dir,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        callbacks.append(tensorboard_callback)

    return callbacks


def train(args, model, ds_train, callbacks):
    """Train model."""
    model.fit(
        ds_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )


def test(args, model, ds_test, callbacks):
    """Test model."""
    model.evaluate(
        ds_test,
        batch_size=args.test_batch_size,
        verbose=2,
        callbacks=callbacks
    )


def main():
    args = parse_args()

    ds_train, ds_test = load_data(
        args, buffer_size=1000, batch_size=args.batch_size)

    model = create_model(args)

    callbacks = get_callbacks(args)

    start = time.time()
    train(args, model, ds_train, callbacks)
    end = time.time()
    print(f"Training time: {end - start:.2f}s")

    test(args, model, ds_test, callbacks)

    if args.save_model:
        model.save('mnist.keras')


if __name__ == "__main__":
    main()
