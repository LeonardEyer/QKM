from multiprocess import Pool
import time
import math
from kernels import K_L_tower, K_layered
import numpy as np
from pennylane import numpy as np
import pennylane as qml
from kernels import *
from tqdm import tqdm


def make_gradient_kernel(kernel):
    def f(params):
        return kernel(params[0], params[1])

    def grad_f(params):
        params = np.array(params, requires_grad=True)
        return qml.jacobian(f)(params)

    return grad_f


def variance(f, params, N):
    return np.var([f(params) for _ in (range(N))])


def variance_parallel(f, params, N):
    with Pool(4) as pool:
        result = pool.map(f, [params for _ in range(N)])
    return np.var(result)


def variance_over_range(f, N, N_repeat, extent=2 * np.pi):
    values = tqdm([(x, y) for x in np.linspace(0, extent, N) for y in np.linspace(0, extent, N)])

    res = np.array(list(map(lambda p: variance_parallel(f, p, N_repeat), values)))
    return np.var(res)


trial_kernels = {
    'XYZ': K_layered,
    'L-prod': K_L_prod,
    'L-tower': K_L_tower,
    'L-cheb': K_L_cheb,
}

if __name__ == "__main__":
    for (label, kernel) in trial_kernels.items():
        k = kernel(8, n_layers=2, qml_backend="qiskit.aer", diff_method="parameter-shift")
        f = make_gradient_kernel(k)
        print(f"Variance of {label} kernel: {variance_over_range(f=f, N=10, N_repeat=10)}")
