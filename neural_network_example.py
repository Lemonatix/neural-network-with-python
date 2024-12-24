import numpy as np

# Data
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary Cross-Entropy
def binary_cross_entropy_loss(A2, Y):
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-9
    return -np.mean(Y * np.log(A2 + epsilon) + (1 - Y) * np.log(1 - A2 + epsilon))

# Network parameters
np.random.seed(1234)
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 100000

for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Loss
    loss = binary_cross_entropy_loss(A2, Y)

    # Backward pass
    # For cross-entropy + sigmoid, dZ2 = (A2 - Y)
    dZ2 = (A2 - Y)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch+1) % 2000 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss:.6f}")

# Final predictions
print("\nTrained network predictions:")
A1 = sigmoid(np.dot(X, W1) + b1)
A2 = sigmoid(np.dot(A1, W2) + b2)
for i in range(len(X)):
    print(f"X={X[i]}, Predicted={A2[i,0]:.4f}, True={Y[i,0]}")
