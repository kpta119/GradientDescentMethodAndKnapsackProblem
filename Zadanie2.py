from autograd import grad
import autograd.numpy as np
from cec2017.functions import f1, f2, f3
import matplotlib.pyplot as plt

def booth(x):
    x1 = x[0]
    x2 = x[1]
    return (x1+2*x2-7)**2+(2*x1+x2-5)**2

def gradientDescentMethod(function, dimensionality=2, beta=0.01):
    arrows = []
    UPPER_BOUND = 100
    x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=dimensionality)
    arrows.append(x)
    grad_fct = grad(function)
    gradient = grad_fct(x)
    while np.linalg.norm(gradient) > 1e-5: 
        gradient = grad_fct(x)
        x = x - beta*gradient
        arrows.append(x)
    return x, arrows


def drawPlot(function):
    _, arrows = gradientDescentMethod(function)
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
            
    plt.contour(X, Y, Z, 20)
    for i in range(len(arrows)-1):
        changeOnAxisX = arrows[i+1][0] - arrows[i][0]
        changeOnAxisY = arrows[i+1][1] - arrows[i][1]
        plt.arrow(arrows[i][0], arrows[i][1], changeOnAxisX, changeOnAxisY, head_width=1, head_length=1, fc='k', ec='k')
    plt.show()
    


#result = gradientDescentMethod(dimensionality=2, function=f1)
result, _ = gradientDescentMethod(dimensionality=2, function=booth)
res = [f"{result[i]:.3f}" for i in range(len(result))]
print(f"Optymalny wynik:", res)
drawPlot(booth)