# Drawing the Sigmoid Function by Matplotlib
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

if __name__ == '__main__':
    values = np.arange(-10, 10, 0.1) # = [-10, -9.9 , ... 9.9] like list we don't count last number 10
    # print(values, type(values))
    plt.plot(values, sigmoid(values)) # sigmoid(array) return array with every element processed by sigmoid function
    # plt.plot(values, [sigmoid(x) for x in values])
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.title('Sigmoid Function in Matplotlib')
    plt.show()