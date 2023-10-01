import numpy as np; np.random.seed(42)
import matplotlib.pyplot as plt

x = [-3, -2, 0, 1.5, 2.2, 3.2, 3.9, 5, 6.75, 9]
y = [7, 7.1, 7.5, 7.7, 8, 8.2, 8.4, 8.8, 9]

X,Y = np.meshgrid(x,y)
Z = np.random.randint(0, 100, size=np.array(X.shape)-1)


fig, ax = plt.subplots()

pc = ax.pcolormesh(X,Y,Z)
fig.colorbar(pc)

def format_coord(x, y):
    xarr = X[0,:]
    yarr = Y[:,0]
    if ((x > xarr.min()) & (x <= xarr.max()) & 
        (y > yarr.min()) & (y <= yarr.max())):
        col = np.searchsorted(xarr, x)-1
        row = np.searchsorted(yarr, y)-1
        z = Z[row, col]
        return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}   [{row},{col}]'
    else:
        return f'x={x:1.4f}, y={y:1.4f}'

ax.format_coord = format_coord

plt.show()