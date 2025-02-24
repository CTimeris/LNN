import numpy as np
import torch


class LNN(torch.nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(LNN, self).__init__()

        # Set LNN hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.reservoir_dim = args.reservoir_dim
        self.leak_rate = args.leak_rate
        self.spectral_radius = args.spectral_radius
        self.num_epochs = args.num_epochs

        # Initialize reservoir weights randomly
        self.reservoir_weights = np.random.randn(self.reservoir_dim, self.reservoir_dim)  # 1000,1000
        # Scale reservoir weights to achieve desired spectral radius
        self.reservoir_weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(self.reservoir_weights)))

        # Initialize input-to-reservoir weights randomly
        self.input_weights = np.random.randn(self.reservoir_dim, self.input_dim)  # 1000, 784

        # Initialize output weights to zero
        self.output_weights = np.zeros((self.reservoir_dim, self.output_dim))  # 1000, 10

    def train(self, input_data, labels):
        num_samples = input_data.shape[0]  # 60000
        reservoir_states = np.zeros((num_samples, self.reservoir_dim))  # 60000, 1000

        for epoch in range(self.num_epochs):
            for i in range(num_samples):
                # Update reservoir state
                if i > 0:
                    reservoir_states[i, :] = (1 - self.leak_rate) * reservoir_states[i - 1, :]
                # np.dot((1000, 784), (1, 784) + np.dot((1000, 1000), (1, 1000))
                reservoir_states[i, :] += self.leak_rate * np.tanh(np.dot(self.input_weights, input_data[i, :]) +
                                                                   np.dot(self.reservoir_weights,
                                                                          reservoir_states[i, :]))

            # Train output weights
            # reservoir_states的伪逆：(1000, 60000), labels: (60000, 10)
            self.output_weights = np.dot(np.linalg.pinv(reservoir_states), labels)  # (1000, 10)

            # Compute training accuracy
            # np.dot((60000, 1000), (1000, 10))
            train_predictions = np.dot(reservoir_states, self.output_weights)  # (60000, 10)
            train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == np.argmax(labels, axis=1))  # 最大值索引位置一样就正确
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Accuracy: {train_accuracy:.4f}")

    def predict_lnn(self, input_data):
        num_samples = input_data.shape[0]  # ~
        reservoir_states = np.zeros((num_samples, self.reservoir_dim))  # ~, 1000

        for i in range(num_samples):
            # Update reservoir state
            if i > 0:
                reservoir_states[i, :] = (1 - self.leak_rate) * reservoir_states[i - 1, :]
            reservoir_states[i, :] += self.leak_rate * np.tanh(np.dot(self.input_weights, input_data[i, :]) +
                                                               np.dot(self.reservoir_weights, reservoir_states[i, :]))

        # Compute predictions using output weights
        predictions = np.dot(reservoir_states, self.output_weights)
        return predictions

    def test(self, x_test, y_test):
        # Evaluate the LNN on test set
        test_predictions = self.predict_lnn(x_test)
        test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
        print(f"Test Accuracy: {test_accuracy:.4f}")


