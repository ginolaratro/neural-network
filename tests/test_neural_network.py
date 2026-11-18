import unittest

from neural_network import NeuralNetwork, sigmoid


class NeuralNetworkTests(unittest.TestCase):
    def test_sigmoid_is_stable(self):
        self.assertEqual(sigmoid(-1000), 0.0)
        self.assertEqual(sigmoid(1000), 1.0)
        self.assertEqual(sigmoid(0), 0.5)

    def test_forward_pass(self):
        network = NeuralNetwork([2, 1], seed=1)
        network.weights = [[[1.0, -1.0]]]
        network.biases = [[0.0]]
        self.assertAlmostEqual(network.predict([1, 1])[0], 0.5)

    def test_learns_xor(self):
        samples = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
        network = NeuralNetwork([2, 4, 1], seed=42)
        losses = network.train(samples, epochs=10000, learning_rate=1.0, seed=42)
        self.assertLess(losses[-1], losses[0])
        for inputs, target in samples:
            self.assertEqual(network.predict(inputs)[0] >= 0.5, bool(target[0]))

    def test_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            NeuralNetwork([2, 0, 1])
        network = NeuralNetwork([2, 1])
        with self.assertRaises(ValueError):
            network.predict([1])
        with self.assertRaises(ValueError):
            network.train([([1, 2], [0, 1])])


if __name__ == "__main__":
    unittest.main()
