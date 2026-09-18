import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    @staticmethod
    def step(x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = np.dot(xi, self.weights) + self.bias
                prediction = self.step(output)
                error = target - prediction
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

    def predict(self, X):
        return self.step(np.dot(X, self.weights) + self.bias)

if __name__ == "__main__":
    X = np.array([[0.9, 0.8], [0.8, 0.7], [0.4, 0.3], [0.3, 0.2]])
    y = np.array([1, 1, 0, 0])
    model = Perceptron()
    model.fit(X, y)
    print("Predictions:", model.predict(X))
