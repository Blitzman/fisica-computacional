import numpy as np
import matplotlib.pyplot as plt

def calculate_loss(y, yHat) :
    """
    Compute the cost for a given set of predictions and the corresponding
    ground truth. The cost is the RMSE (Root Mean Squared Error).
    """
    n = len(yHat)
    loss = 1.0 / n * np.sum(np.square(yHat - y))
    return loss

def gradient_descent(x, y, weights, alpha, iterations) :
    """
    TODO: comment
    """
    n = len(y)

    loss_per_iteration_ = np.zeros(iterations)

    for i in range(iterations) :

      y_hat_ = np.dot(x, weights)

if __name__ == "__main__" :

    learning_rate_ = 0.01
    iterations_ = 10

    x_ = np.random.rand(100, 1)
    y_ = 4.0 + 3.0 * x_ + np.random.rand(100, 1)

    plt.plot(x_, y_)
    plt.show()

    weights_ = np.random.random((2, 1))
    print(weights_)

    gradient_descent()