import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def plot_activations(output="results/activation_functions.png"):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-10, 10, 500)
    plt.figure(figsize=(10, 6))
    plt.plot(x, sigmoid(x), label="Sigmoid")
    plt.plot(x, tanh(x), label="Tanh")
    plt.plot(x, relu(x), label="ReLU")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Activation Functions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

if __name__ == "__main__":
    plot_activations()
    print("Saved results/activation_functions.png")
