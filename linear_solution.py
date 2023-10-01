import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 生成数据
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)

X1, X2 = np.meshgrid(x1, x2)
Y = X1 + 2*X2
Y_sigmoid = sigmoid(Y)

# 创建3D绘图
fig = plt.figure()

# 第一个子图
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis')
ax.set_xlabel('X1 Axis')
ax.set_ylabel('X2 Axis')
ax.set_zlabel('Y Axis')
ax.set_title('3D plot of y = x1 + 2*x2')

# 第二个子图
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.plot_surface(X1, X2, Y_sigmoid, cmap='viridis')
ax1.set_xlabel('X1 Axis')
ax1.set_ylabel('X2 Axis')
ax1.set_zlabel('Y Axis')
ax1.set_title('3D plot of sigmoid(y)')

plt.tight_layout()  # 使子图之间的间距更紧密
plt.show()
