import numpy as np
import tensorflow as tf
from tensorflow import keras


def initialize_weights(input_dim, reservoir_dim, output_dim, spectral_radius):
    # Initialize reservoir weights randomly
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)   # 1000,1000
    # Scale reservoir weights to achieve desired spectral radius
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))

    # Initialize input-to-reservoir weights randomly
    input_weights = np.random.randn(reservoir_dim, input_dim)       # 1000, 784

    # Initialize output weights to zero
    output_weights = np.zeros((reservoir_dim, output_dim))          # 1000, 10

    return reservoir_weights, input_weights, output_weights


def train_lnn(input_data, labels, reservoir_weights, input_weights, output_weights, leak_rate, num_epochs):
    num_samples = input_data.shape[0]   # 60000
    reservoir_dim = reservoir_weights.shape[0]  # 1000
    reservoir_states = np.zeros((num_samples, reservoir_dim))   # 60000, 1000

    for epoch in range(num_epochs):
        for i in range(num_samples):
            # Update reservoir state
            if i > 0:
                reservoir_states[i, :] = (1 - leak_rate) * reservoir_states[i - 1, :]
            # np.dot((1000, 784), (1, 784) + np.dot((1000, 1000), (1, 1000))
            reservoir_states[i, :] += leak_rate * np.tanh(np.dot(input_weights, input_data[i, :]) +
                                                          np.dot(reservoir_weights, reservoir_states[i, :]))

        # Train output weights
        # reservoir_states的伪逆：(1000, 60000), labels: (60000, 10)
        output_weights = np.dot(np.linalg.pinv(reservoir_states), labels)  # (1000, 10)

        # Compute training accuracy
        # np.dot((60000, 1000), (1000, 10))
        train_predictions = np.dot(reservoir_states, output_weights)    # (60000, 10)
        train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == np.argmax(labels, axis=1))  # 最大值索引位置一样就正确
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}")

    return output_weights   # （1000， 10）


def predict_lnn(input_data, reservoir_weights, input_weights, output_weights, leak_rate):
    num_samples = input_data.shape[0]       # ~
    reservoir_dim = reservoir_weights.shape[0]  # 1000
    reservoir_states = np.zeros((num_samples, reservoir_dim))   # ~, 1000

    for i in range(num_samples):
        # Update reservoir state
        if i > 0:
            reservoir_states[i, :] = (1 - leak_rate) * reservoir_states[i - 1, :]
        reservoir_states[i, :] += leak_rate * np.tanh(np.dot(input_weights, input_data[i, :]) +
                                                      np.dot(reservoir_weights, reservoir_states[i, :]))

    # Compute predictions using output weights
    predictions = np.dot(reservoir_states, output_weights)
    return predictions


# 加载保存的 MNIST 数据
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
print("数据加载完成！")

x_train = x_train.reshape((60000, 784)) / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
x_test = x_test.reshape((10000, 784)) / 255.0

print("input data size: {0}, {1}, {2}, {3}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))


# Set LNN hyperparameters
input_dim = 784
reservoir_dim = 1000
output_dim = 10
leak_rate = 0.1
spectral_radius = 0.9
num_epochs = 10


# Initialize LNN weights
reservoir_weights, input_weights, output_weights = initialize_weights(input_dim, reservoir_dim, output_dim, spectral_radius)

# Train the LNN
output_weights = train_lnn(x_train, y_train, reservoir_weights, input_weights, output_weights, leak_rate, num_epochs)

# Evaluate the LNN on test set
test_predictions = predict_lnn(x_test, reservoir_weights, input_weights, output_weights, leak_rate)
test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {test_accuracy:.4f}")
