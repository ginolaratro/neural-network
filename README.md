# Neural network from scratch

A small feed-forward neural network written in pure Python, without machine learning libraries. It learns using sigmoid activations, mean squared error, backpropagation, and stochastic gradient descent. The included example learns XOR, a problem that requires a hidden layer.

## Run

Requires Python 3. Run the example from the project directory:

```sh
python3 neural_network.py
```

It prints the training loss and predicted probability for each XOR input. Run the tests with:

```sh
python3 -m unittest discover -s tests -v
```

## Use

```python
from neural_network import NeuralNetwork

samples = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
network = NeuralNetwork([2, 4, 1], seed=42)
losses = network.train(samples, epochs=10000, learning_rate=1.0, seed=42)
print(network.predict([0, 1]))
```

The layer sizes include the input and output layers. Each target must have the same length as the output layer. Outputs are between 0 and 1. The returned loss history is the average training loss observed during each epoch, as weights are updated. Seeds make initialization and training order reproducible. This educational implementation prioritizes clarity over performance and is not intended for large datasets.
