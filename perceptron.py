import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Generate the dataset
np.random.seed(42)
X = np.random.rand(100, 2)
y = (X[:, 1] > X[:, 0] + 0.1).astype(int)

# Step 2: Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Step 3: Define the Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)

                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# Step 4: Train and test the perceptron
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron(learning_rate=0.1, epochs=1000)
perceptron.fit(X_train, y_train)

predictions = perceptron.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 5: Visualize the decision boundary
x0 = np.linspace(0, 1, 100)
x1 = -(perceptron.weights[0] * x0 + perceptron.bias) / perceptron.weights[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', label="Data")
plt.plot(x0, x1, color="black", label="Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
