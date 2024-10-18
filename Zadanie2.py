from autograd import grad
import autograd.numpy as np
from cec2017.functions import f1, f2, f3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def booth(x):
    x1 = x[0]
    x2 = x[1]
    return (x1+2*x2-7)**2+(2*x1+x2-5)**2


def gradientDescentMethod(function, maxIterations, dimensionality=10, beta=0.00000001):
    UPPER_BOUND = 100
    x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=dimensionality)
    grad_fct = grad(function)
    gradient = grad_fct(x)
    iteration = 0
    while np.linalg.norm(gradient) > 1e-5:
        xPrevious = x
        gradient = grad_fct(x)
        x = x - beta*gradient
        x = np.clip(x, -UPPER_BOUND, UPPER_BOUND)
        changeOnAxisX = x[0] - xPrevious[0]
        changeOnAxisY = x[1] - xPrevious[1]
        plt.arrow(xPrevious[0],xPrevious[1],changeOnAxisX, changeOnAxisY, head_width=1, head_length=1, fc='k', ec='k')
        iteration += 1
        if iteration > maxIterations:
            break
    return x

def drawPlot(function):
    MAX_X = 100
    PLOT_STEP = 0.1

    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    q=function
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = q(np.array([X[i, j], Y[i, j]]))
            
    contour = plt.contour(X, Y, Z, 30)
    fmt = ticker.FormatStrFormatter("%.1e")
    plt.clabel(contour, inline=True, fontsize=6, fmt=fmt)

    plt.title(f"Function: {function.__name__}" )
    plt.xlabel("x1")
    plt.ylabel("x2")




def main():
    drawPlot(f3)
    gradientDescentMethod(f3, 10000)
    plt.show()

if __name__ == "__main__":
    main()