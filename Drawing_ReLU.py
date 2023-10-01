# Drawing the ReLU Function by Matplotlib
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return max(0.0, x) # 回傳0 or x 之間最大的數字
def npReLU(x):
    return np.maximum(0,x_coordinate) # 回傳0 or x 之間最大的數字 for numpy

if __name__ == '__main__':

    # Method I : Not using package for HackerRank
    x_coordinate = [x for x in range(-10,10)] # 印出 x = -10 ~ 10的數字至 list 中, interval = 1 (只能設置int 為 interval)
    print(x_coordinate)

    y_value = [ReLU(x_element) for x_element in x_coordinate]
    print(y_value)

    # Method II : Using package : Numpy Array that can prcoess data easier
    x_coordinate = np.arange(-10,10,0.1) # -10 至 9 之間，interval = 0.1
    print(x_coordinate)
    y_value = npReLU(x_coordinate)
    print(y_value)
    print(y_value.shape)

    plt.xlabel('x')
    plt.ylabel('ReLU(x)')
    plt.title('Rectified Linear Unit Funciton in Matplotlib')
    plt.plot(x_coordinate,y_value)
    plt.show()


