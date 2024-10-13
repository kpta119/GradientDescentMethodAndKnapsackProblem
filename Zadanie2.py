from autograd import grad
import autograd.numpy as np

def f1(x):
    return x**2

def f2(x):
    x1 = x[0]
    x2 = x[1]
    return x1**2 + x2**2 -x1*x2

def gradientDescentMethod(dimensionality, function, beta=0.1):
    UPPER_BOUND = 100
    x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=dimensionality)
    grad_fct = grad(function)
    gradient = grad_fct(x)
    while np.linalg.norm(gradient) > 1e-5: 
        gradient = grad_fct(x)
        x = x - beta*gradient
    return x

result = gradientDescentMethod(dimensionality=2, function=f2)
res = [f"{result[i]:.3f}" for i in range(len(result))]
print(f"Optymalny wynik:", res)
