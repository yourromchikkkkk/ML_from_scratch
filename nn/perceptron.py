class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.01):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(weight * x for weight, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0
    
    def train(self, training_data, epochs=1):
        for epoch in range(epochs):
            errors = 0
            for x, y in training_data:
                prediction = self.predict(x)
                error = y - prediction
                if error != 0:
                    errors += 1
                    for idx in range(len(self.weights)):
                        self.weights[idx] += self.lr * error * x[idx]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")

def test_perceptron():
    def test_gate(name, n_inputs, data):
        print(f"=== {name} ===")
        p = Perceptron(n_inputs)
        p.train(data, epochs=100)
        print(f"  Weights: {p.weights}, Bias: {p.bias}")
        for inputs, expected in data:
            result = p.predict(inputs)
            status = "OK" if result == expected else "WRONG"
            print(f"  {inputs} -> {result} (expected {expected}) {status}")
        print()


    and_data = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
    ]

    or_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1),
    ]

    not_data = [
        ([0], 1),
        ([1], 0),
    ]

    xor_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    test_gate("AND Gate", 2, and_data)
    test_gate("OR Gate", 2, or_data)
    test_gate("NOT Gate", 1, not_data)

    print("=== XOR Gate (single perceptron - will fail) ===")
    p_xor = Perceptron(2)
    p_xor.train(xor_data, epochs=1000)
    for inputs, expected in xor_data:
        result = p_xor.predict(inputs)
        status = "OK" if result == expected else "WRONG"
        print(f"  {inputs} -> {result} (expected {expected}) {status}")
    print()


    def xor_network(x1, x2):
        or_neuron = Perceptron(2)
        or_neuron.weights = [1.0, 1.0]
        or_neuron.bias = -0.5

        nand_neuron = Perceptron(2)
        nand_neuron.weights = [-1.0, -1.0]
        nand_neuron.bias = 1.5

        and_neuron = Perceptron(2)
        and_neuron.weights = [1.0, 1.0]
        and_neuron.bias = -1.5

        hidden1 = or_neuron.predict([x1, x2])
        hidden2 = nand_neuron.predict([x1, x2])
        return and_neuron.predict([hidden1, hidden2])


    print("=== XOR Gate (multi-layer network - works) ===")
    for inputs, expected in xor_data:
        result = xor_network(inputs[0], inputs[1])
        status = "OK" if result == expected else "WRONG"
        print(f"  {inputs} -> {result} (expected {expected}) {status}")
    print()

if __name__ == "__main__":
    test_perceptron()