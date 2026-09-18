import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def run_demo(epochs=1000, learning_rate=0.1):
    np.random.seed(42)
    X = np.array([
        [0.90, 0.80, 0.85, 0.88],
        [0.70, 0.60, 0.65, 0.62],
        [0.50, 0.45, 0.48, 0.50],
        [0.30, 0.25, 0.35, 0.32]
    ])
    y = np.array([3, 2, 1, 0])
    Y = np.eye(4)[y]

    W1 = np.random.randn(4, 8) * 0.1
    b1 = np.zeros((1, 8))
    W2 = np.random.randn(8, 4) * 0.1
    b2 = np.zeros((1, 4))
    losses = []

    for _ in range(epochs):
        Z1 = X @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        loss = -np.mean(np.sum(Y * np.log(A2 + 1e-8), axis=1))
        losses.append(loss)

        dZ2 = A2 - Y
        dW2 = A1.T @ dZ2 / len(X)
        db2 = np.mean(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X.T @ dZ1 / len(X)
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Manual 2-Layer Neural Network Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/manual_nn_loss.png", dpi=150)
    plt.close()

    return np.argmax(A2, axis=1), losses

if __name__ == "__main__":
    pred, losses = run_demo()
    print("Predicted classes:", pred)
    print("Final loss:", losses[-1])
