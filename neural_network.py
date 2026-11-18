"""A small fully connected neural network implemented with Python's standard library."""

import math
import random


def sigmoid(value):
    """Compute the sigmoid without overflowing for large negative inputs."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class NeuralNetwork:
    """Dense sigmoid layers trained with stochastic gradient descent and mean squared error."""

    def __init__(self, layers, seed=None):
        layers = list(layers)
        if len(layers) < 2 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in layers):
            raise ValueError("layers must contain at least two positive integers")
        self.layers = layers
        rng = random.Random(seed)
        self.weights = [
            [[rng.uniform(-math.sqrt(6 / (inputs + outputs)), math.sqrt(6 / (inputs + outputs)))
              for _ in range(inputs)] for _ in range(outputs)]
            for inputs, outputs in zip(layers, layers[1:])
        ]
        self.biases = [[0.0] * size for size in layers[1:]]

    def _forward(self, inputs):
        activations = [list(inputs)]
        if len(activations[0]) != self.layers[0]:
            raise ValueError(f"expected {self.layers[0]} inputs")
        for weights, biases in zip(self.weights, self.biases):
            previous = activations[-1]
            activations.append([sigmoid(sum(w * x for w, x in zip(row, previous)) + bias)
                                for row, bias in zip(weights, biases)])
        return activations

    def predict(self, inputs):
        return self._forward(inputs)[-1]

    def train(self, samples, epochs=1000, learning_rate=0.5, seed=None):
        """Train on (input, target) pairs and return mean squared error per epoch."""
        samples = [(list(inputs), list(target)) for inputs, target in samples]
        if not samples:
            raise ValueError("samples must not be empty")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        for inputs, target in samples:
            if len(inputs) != self.layers[0] or len(target) != self.layers[-1]:
                raise ValueError("sample dimensions must match the network")
        rng = random.Random(seed)
        history = []
        for _ in range(epochs):
            rng.shuffle(samples)
            total_loss = 0.0
            for inputs, target in samples:
                activations = self._forward(inputs)
                output = activations[-1]
                total_loss += sum((actual - expected) ** 2 for actual, expected in zip(output, target)) / len(target)
                deltas = [None] * len(self.weights)
                deltas[-1] = [2 * (actual - expected) * actual * (1 - actual) / len(target)
                              for actual, expected in zip(output, target)]
                for layer in range(len(deltas) - 2, -1, -1):
                    deltas[layer] = [
                        sum(self.weights[layer + 1][j][i] * deltas[layer + 1][j]
                            for j in range(len(deltas[layer + 1]))) * activation * (1 - activation)
                        for i, activation in enumerate(activations[layer + 1])
                    ]
                for layer, delta in enumerate(deltas):
                    for neuron, error in enumerate(delta):
                        for i, activation in enumerate(activations[layer]):
                            self.weights[layer][neuron][i] -= learning_rate * error * activation
                        self.biases[layer][neuron] -= learning_rate * error
            history.append(total_loss / len(samples))
        return history


if __name__ == "__main__":
    xor = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    network = NeuralNetwork([2, 4, 1], seed=42)
    losses = network.train(xor, epochs=10000, learning_rate=1.0, seed=42)
    print(f"Training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    for inputs, target in xor:
        print(f"{inputs} -> {network.predict(inputs)[0]:.4f} (expected {target[0]})")
