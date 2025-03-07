<<<<<<< HEAD
# cec2017.composition
# Author: Duncan Tilley
# Composition function definitions, f21 to f30

from . import basic
from . import transforms
from . import hybrid

import numpy as np

def _calc_w(x, sigma):
    nx = len(x)
    w = 0
    for i in range(0, nx):
        w += x[i]*x[i]
    if (w != 0):
        w = ((1.0/w)**0.5) * np.exp(-w / (2.0*nx*sigma*sigma))
    else:
        w = float('inf')
    return w

def f21(x, rotations=None, shifts=None):
    """
    Composition Function 1 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][0]
    if shifts is None:
        shifts = transforms.shifts_cf[0]

    N = 3
    funcs = [basic.rosenbrock, basic.high_conditioned_elliptic, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0])
    lambdas = np.array([1.0, 1.0e-6, 1.0])
    biases = np.array([0.0, 100.0, 200.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2100

def f22(x, rotations=None, shifts=None):
    """
    Composition Function 2 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][1]
    if shifts is None:
        shifts = transforms.shifts_cf[1]

    N = 3
    funcs = [basic.rastrigin, basic.griewank, basic.modified_schwefel]
    sigmas = np.array([10.0, 20.0, 30.0])
    lambdas = np.array([1.0, 10.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2200

def f23(x, rotations=None, shifts=None):
    """
    Composition Function 3 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][2]
    if shifts is None:
        shifts = transforms.shifts_cf[2]

    N = 4
    funcs = [basic.rosenbrock, basic.ackley, basic.modified_schwefel, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0])
    lambdas = np.array([1.0, 10.0, 1.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2300

def f24(x, rotations=None, shifts=None):
    """
    Composition Function 4 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][3]
    if shifts is None:
        shifts = transforms.shifts_cf[3]

    N = 4
    funcs = [basic.ackley, basic.high_conditioned_elliptic, basic.griewank, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0])
    lambdas = np.array([1.0, 1.0e-6, 10.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2400

def f25(x, rotations=None, shifts=None):
    """
    Composition Function 5 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][4]
    if shifts is None:
        shifts = transforms.shifts_cf[4]

    N = 5
    funcs = [basic.rastrigin, basic.happy_cat, basic.ackley, basic.discus, basic.rosenbrock]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lambdas = np.array([10.0, 1.0, 10.0, 1.0e-6, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2500

def f26(x, rotations=None, shifts=None):
    """
    Composition Function 6 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][5]
    if shifts is None:
        shifts = transforms.shifts_cf[5]

    N = 5
    funcs = [basic.expanded_schaffers_f6, basic.modified_schwefel, basic.griewank, basic.rosenbrock, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 20.0, 30.0, 40.0])
    # Note: the lambdas specified in the problem definitions (below) differ from what is used in the code
    #lambdas = np.array([1.0e-26, 10.0, 1.0e-6, 10.0, 5.0e-4])
    lambdas = np.array([5.0e-4, 1.0, 10.0, 1.0, 10.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2600

def f27(x, rotations=None, shifts=None):
    """
    Composition Function 7 (N=6)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][6]
    if shifts is None:
        shifts = transforms.shifts_cf[6]

    N = 6
    funcs = [
        basic.h_g_bat,
        basic.rastrigin,
        basic.modified_schwefel,
        basic.bent_cigar,
        basic.high_conditioned_elliptic,
        basic.expanded_schaffers_f6]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lambdas = np.array([10.0, 10.0, 2.5, 1.0e-26, 1.0e-6, 5.0e-4])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2700

def f28(x, rotations=None, shifts=None):
    """
    Composition Function 8 (N=6)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][7]
    if shifts is None:
        shifts = transforms.shifts_cf[7]

    N = 6
    funcs = [
        basic.ackley,
        basic.griewank,
        basic.discus,
        basic.rosenbrock,
        basic.happy_cat,
        basic.expanded_schaffers_f6]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lambdas = np.array([10.0, 10.0, 1.0e-6, 1.0, 1.0, 5.0e-4])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2800

def f29(x, rotations=None, shifts=None, shuffles=None):
    """
    Composition Function 9 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
        shuffles (array): Optional shuffle vectors (NxD). If None (default), the
            official permutation vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][8]
    if shifts is None:
        shifts = transforms.shifts_cf[8]
    if shuffles is None:
        shuffles = transforms.shuffles_cf[nx][0]

    N = 3
    funcs = [hybrid.f15, hybrid.f16, hybrid.f17]
    sigmas = np.array([10.0, 30.0, 50.0])
    biases = np.array([0.0, 100.0, 200.0])
    offsets = np.array([1500, 1600, 1700]) # subtract F* added at the end of the functions
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](x, rotation=rotations[i], shift=shifts[i][:nx], shuffle=shuffles[i])
        vals[i] -= offsets[i]
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (vals + biases)) + 2900

def f30(x, rotations=None, shifts=None, shuffles=None):
    """
    Composition Function 10 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
        shuffles (array): Optional shuffle vectors (NxD). If None (default), the
            official permutation vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][9]
    if shifts is None:
        shifts = transforms.shifts_cf[9]
    if shuffles is None:
        shuffles = transforms.shuffles_cf[nx][1]

    N = 3
    funcs = [hybrid.f15, hybrid.f18, hybrid.f19]
    sigmas = np.array([10.0, 30.0, 50.0])
    biases = np.array([0.0, 100.0, 200.0])
    offsets = np.array([1500, 1800, 1900]) # subtract F* added at the end of the functions
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](x, rotation=rotations[i], shift=shifts[i][:nx], shuffle=shuffles[i])
        vals[i] -= offsets[i]
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (vals + biases)) + 3000
=======
# cec2017.composition
# Author: Duncan Tilley
# Composition function definitions, f21 to f30

from . import basic
from . import transforms
from . import hybrid

import numpy as np

def _calc_w(x, sigma):
    nx = len(x)
    w = 0
    for i in range(0, nx):
        w += x[i]*x[i]
    if (w != 0):
        w = ((1.0/w)**0.5) * np.exp(-w / (2.0*nx*sigma*sigma))
    else:
        w = float('inf')
    return w

def f21(x, rotations=None, shifts=None):
    """
    Composition Function 1 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][0]
    if shifts is None:
        shifts = transforms.shifts_cf[0]

    N = 3
    funcs = [basic.rosenbrock, basic.high_conditioned_elliptic, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0])
    lambdas = np.array([1.0, 1.0e-6, 1.0])
    biases = np.array([0.0, 100.0, 200.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2100

def f22(x, rotations=None, shifts=None):
    """
    Composition Function 2 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][1]
    if shifts is None:
        shifts = transforms.shifts_cf[1]

    N = 3
    funcs = [basic.rastrigin, basic.griewank, basic.modified_schwefel]
    sigmas = np.array([10.0, 20.0, 30.0])
    lambdas = np.array([1.0, 10.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2200

def f23(x, rotations=None, shifts=None):
    """
    Composition Function 3 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][2]
    if shifts is None:
        shifts = transforms.shifts_cf[2]

    N = 4
    funcs = [basic.rosenbrock, basic.ackley, basic.modified_schwefel, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0])
    lambdas = np.array([1.0, 10.0, 1.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2300

def f24(x, rotations=None, shifts=None):
    """
    Composition Function 4 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][3]
    if shifts is None:
        shifts = transforms.shifts_cf[3]

    N = 4
    funcs = [basic.ackley, basic.high_conditioned_elliptic, basic.griewank, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0])
    lambdas = np.array([1.0, 1.0e-6, 10.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2400

def f25(x, rotations=None, shifts=None):
    """
    Composition Function 5 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][4]
    if shifts is None:
        shifts = transforms.shifts_cf[4]

    N = 5
    funcs = [basic.rastrigin, basic.happy_cat, basic.ackley, basic.discus, basic.rosenbrock]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lambdas = np.array([10.0, 1.0, 10.0, 1.0e-6, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2500

def f26(x, rotations=None, shifts=None):
    """
    Composition Function 6 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][5]
    if shifts is None:
        shifts = transforms.shifts_cf[5]

    N = 5
    funcs = [basic.expanded_schaffers_f6, basic.modified_schwefel, basic.griewank, basic.rosenbrock, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 20.0, 30.0, 40.0])
    # Note: the lambdas specified in the problem definitions (below) differ from what is used in the code
    #lambdas = np.array([1.0e-26, 10.0, 1.0e-6, 10.0, 5.0e-4])
    lambdas = np.array([5.0e-4, 1.0, 10.0, 1.0, 10.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2600

def f27(x, rotations=None, shifts=None):
    """
    Composition Function 7 (N=6)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][6]
    if shifts is None:
        shifts = transforms.shifts_cf[6]

    N = 6
    funcs = [
        basic.h_g_bat,
        basic.rastrigin,
        basic.modified_schwefel,
        basic.bent_cigar,
        basic.high_conditioned_elliptic,
        basic.expanded_schaffers_f6]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lambdas = np.array([10.0, 10.0, 2.5, 1.0e-26, 1.0e-6, 5.0e-4])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2700

def f28(x, rotations=None, shifts=None):
    """
    Composition Function 8 (N=6)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][7]
    if shifts is None:
        shifts = transforms.shifts_cf[7]

    N = 6
    funcs = [
        basic.ackley,
        basic.griewank,
        basic.discus,
        basic.rosenbrock,
        basic.happy_cat,
        basic.expanded_schaffers_f6]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lambdas = np.array([10.0, 10.0, 1.0e-6, 1.0, 1.0, 5.0e-4])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2800

def f29(x, rotations=None, shifts=None, shuffles=None):
    """
    Composition Function 9 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
        shuffles (array): Optional shuffle vectors (NxD). If None (default), the
            official permutation vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][8]
    if shifts is None:
        shifts = transforms.shifts_cf[8]
    if shuffles is None:
        shuffles = transforms.shuffles_cf[nx][0]

    N = 3
    funcs = [hybrid.f15, hybrid.f16, hybrid.f17]
    sigmas = np.array([10.0, 30.0, 50.0])
    biases = np.array([0.0, 100.0, 200.0])
    offsets = np.array([1500, 1600, 1700]) # subtract F* added at the end of the functions
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](x, rotation=rotations[i], shift=shifts[i][:nx], shuffle=shuffles[i])
        vals[i] -= offsets[i]
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (vals + biases)) + 2900

def f30(x, rotations=None, shifts=None, shuffles=None):
    """
    Composition Function 10 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
        shuffles (array): Optional shuffle vectors (NxD). If None (default), the
            official permutation vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][9]
    if shifts is None:
        shifts = transforms.shifts_cf[9]
    if shuffles is None:
        shuffles = transforms.shuffles_cf[nx][1]

    N = 3
    funcs = [hybrid.f15, hybrid.f18, hybrid.f19]
    sigmas = np.array([10.0, 30.0, 50.0])
    biases = np.array([0.0, 100.0, 200.0])
    offsets = np.array([1500, 1800, 1900]) # subtract F* added at the end of the functions
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](x, rotation=rotations[i], shift=shifts[i][:nx], shuffle=shuffles[i])
        vals[i] -= offsets[i]
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (vals + biases)) + 3000
>>>>>>> df1770f47ca173838b6727de9080ec3646b50342
