import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_pred):
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

def sgd_update(weights, gradients, learning_rate=0.01):
    return weights - learning_rate * gradients

class Adam:
    """Educational Adam implementation for a single parameter array."""
    def __init__(self, shape, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def update(self, weights, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

if __name__ == "__main__":
    actual = np.array([80, 90, 70])
    predicted = np.array([78, 87, 74])
    print("MSE:", mse(actual, predicted))

    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
    print("Cross Entropy:", cross_entropy(y_true, y_pred))
