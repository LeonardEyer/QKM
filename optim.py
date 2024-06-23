from pennylane import numpy as np
import pennylane as qml
from itertools import product
from util import get_best_params, gram_generator
from sklearn.svm import SVR
import scipy


def mmr(X, Y, F, K, method='Newton-CG'):
    x_data = np.array(X)
    y_data = np.array(Y)
    f_data = np.array(F)

    # Evaluate the kernel at the data points
    k_evaluated_at_data_points = np.zeros((len(x_data), len(y_data)))
    for i in range(len(x_data)):
        for j in range(len(x_data)):
            k_evaluated_at_data_points[i, j] = K(x_data[i], y_data[j])

    def trial_function_evaluated_at_data_points(a, b):
        result = np.zeros_like(a)
        result += b
        for i in range(len(a)):
            result += a[i] * k_evaluated_at_data_points[:, i]
        return result

    def trial_function(x, a, b):
        result = b
        for i in range(len(a)):
            result += a[i] * K(x, y_data[i])
        return result

    def loss_function(alphas):
        b = alphas[0]
        a = alphas[1:]
        f_trial = trial_function_evaluated_at_data_points(a, b)
        result = np.sum((f_data - f_trial) ** 2)
        return result

    def grad_function(alphas):
        alphas = np.array(alphas, requires_grad=True)
        return qml.jacobian(loss_function)(alphas)

    losses = []
    def collect_losses_callback(intermediate_result):
        losses.append(intermediate_result.fun)

    x0 = np.array([np.mean(f_data), *np.zeros_like(y_data)])

    res = scipy.optimize.minimize(loss_function, x0=x0, jac=grad_function, method=method,
                                  callback=collect_losses_callback, options={'disp': True})

    b = res.x[0]
    a = res.x[1:]

    def optimized_trial_function(x):
        return trial_function(x, a, b)

    return optimized_trial_function, losses


def mmr_DE(DE, data, x0, f0, k, method='Newton-CG'):
    x_data = np.array(data[0])
    y_data = np.array(data[1])

    k_at_initial_condition = np.zeros(len(x_data))
    for i in range(len(x_data)):
        k_at_initial_condition[i] = k(x0, y_data[i])

    def f0_trial(a, b):
        result = 0
        result += b
        for i in range(len(a)):
            result += a[i] * k_at_initial_condition[i]
        return result

    def K_grad(x, y):
        params = np.array([x, y], requires_grad=True)
        return qml.jacobian(k, argnum=[0])(params[0], params[1])[0]

    k_evaluated_at_data_points = np.zeros((len(x_data), len(x_data)))
    dk_dx_evaluated_at_data_points = np.zeros((len(x_data), len(x_data)))
    for i in range(len(x_data)):
        for j in range(len(x_data)):

            x_i = x_data[i]
            x_j = x_data[j]

            dk_dx_evaluated_at_data_points[i, j] = K_grad(x_i, x_j)
            k_evaluated_at_data_points[i, j] = k(x_i, x_j)

    def trial_function_evaluated_at_data_points(a, b):
        result = np.zeros_like(a)
        result += b
        for i in range(len(a)):
            result += a[i] * k_evaluated_at_data_points[:, i]
        return result

    def dx_trial_function_evaluated_at_data_points(a, b):
        result = np.zeros_like(a)
        for i in range(len(a)):
            result += a[i] * dk_dx_evaluated_at_data_points[:, i]
        return result

    def trial_function(x, a, b):
        result = b
        for i in range(len(a)):
            result += a[i] * k(x, y_data[i])
        return result

    def loss_function(alphas):
        b = alphas[0]
        a = alphas[1:]
        result = (f0 - f0_trial(a, b)) ** 2
        f_trial = trial_function_evaluated_at_data_points(a, b)
        df_dx_trial = dx_trial_function_evaluated_at_data_points(a, b)
        for i in range(len(x_data)):
            result += DE(x_data[i], f_trial[i], df_dx_trial[i]) ** 2
        return result


    losses = []
    def collect_losses_callback(intermediate_result):
        losses.append(intermediate_result.fun)
        if len(losses) % 5 == 0:
            print("Loss: ", intermediate_result.fun)

    def grad_function(alphas):
        alphas = np.array(alphas, requires_grad=True)
        return qml.jacobian(loss_function)(alphas)

    x0 = np.array([np.sum(y_data) / len(y_data), *np.zeros_like(y_data)], requires_grad=True)
    res = scipy.optimize.minimize(loss_function, x0, jac=grad_function, method=method,
                                  callback=collect_losses_callback, options={'disp': True}).x

    b = res[0]
    a = res[1:]

    def optimized_trial_function(x):
        return trial_function(x, a, b)

    return optimized_trial_function


def multivalued_mmr_DE(DE, data, x0, f0, K, IC_weight=1, method='Newton-CG'):
    x_data = np.array(data[0])
    y_data = np.array(data[1])
    x0 = np.array(x0)
    f0 = np.array(f0)
    # f: R^n -> R^m
    # kernel evaluation points: n_x and n_y points in R^n / R^m respectively
    n = len(x0)
    m = len(f0)
    n_x = len(x_data)
    n_y = len(y_data)

    a0 = np.array([*np.sum(y_data, axis=0).reshape(y_data.shape[1]) / y_data.shape[0], *np.zeros(m * n * n_y * m)],
                  requires_grad=True)

    def K_grad(x, y):
        params = np.array([x, y], requires_grad=True)
        return qml.jacobian(K, argnum=[0])(params[0], params[1])

    k_at_initial_condition = np.zeros((n, n_y, m))
    for (i, j, k) in product(range(n), range(n_y), range(m)):
        k_at_initial_condition[i, j, k] = K(x0[i], y_data[j, k])

    def f0_trial(a, b):
        result = np.array([b[i] + sum([a[i, j, k, l] * k_at_initial_condition[j, k, l]
                                       for j, k, l in product(range(n), range(n_y), range(m))])
                           for i in range(m)])
        return result

    k_evaluated_at_data_points = np.zeros((n_x, n, n_y, m))
    dk_dx_evaluated_at_data_points = np.zeros((n_x, n, n_y, m, n))
    for i, j, k, l in product(range(n_x), range(n), range(n_y), range(m)):
        x_ij = x_data[i, j]
        dk_dx_evaluated_at_data_points[i, j, k, l] = K_grad(x_ij, y_data[k, l])
        k_evaluated_at_data_points[i, j, k, l] = K(x_ij, y_data[k, l])

    def trial_function_evaluated_at_data_points(a, b):
        result = np.array([])
        for i, j in product(range(n_x), range(m)):
            result = np.append(result,
                               np.array([b[j] + sum([a[j, k, l, o] * k_evaluated_at_data_points[i, k, l, o]
                                                     for k, l, o in product(range(n), range(n_y), range(m))])]))
        return result.reshape(n_x, m)

    def dx_trial_function_evaluated_at_data_points(a, b):
        # returns df/dx_1 in first element, df/dx_2 in second element etc.
        result = np.array([])
        for q in range(n):
            result_q = np.array([])
            for i, j in product(range(n_x), range(m)):
                result_q = np.append(result_q,
                                     np.array([sum([a[j, k, l, o] * dk_dx_evaluated_at_data_points[i, k, l, o, q]
                                                    for k, l, o in product(range(n), range(n_y), range(m))])]))
            result = np.append(result, result_q.reshape(n_x, m))
        return result.reshape(n_x, m)

    def trial_function(x, a, b):
        result = np.array([b[i] + sum([a[i, j, k, l] * K(x[j], y_data[k, l])
                                       for j, k, l in product(range(n), range(n_y), range(m))]) for i in range(m)])
        return result

    def loss_function(alphas):
        b = alphas[:m]
        a = np.reshape(alphas[m:], (m, n, n_y, m))
        result = IC_weight * np.linalg.norm((f0 - f0_trial(a, b))) ** 2
        f_trial = trial_function_evaluated_at_data_points(a, b)
        df_dx_trial = dx_trial_function_evaluated_at_data_points(a, b)
        for i in range(len(x_data)):
            result += np.linalg.norm(DE(x_data[i], f_trial[i], df_dx_trial[i])) ** 2
        return result

    # return loss_function
    def grad_function(alphas):
        alphas = np.array(alphas, requires_grad=True)
        return qml.jacobian(loss_function)(alphas)

    def print_loss_callback(intermediate_result):
        print("Loss: ", intermediate_result.fun)

    minimization = scipy.optimize.minimize(
        loss_function, a0, jac=grad_function, method=method, callback=print_loss_callback, options={'disp': True}).x

    b = minimization[:m]
    a = np.array(minimization[m:])
    a = np.reshape(a, (m, n, n_y, m))

    def optimized_trial_function(x):
        return trial_function(x, a, b)

    # return loss_function
    return optimized_trial_function


def SVR_fit(X, Y, K, params=None):
    K = gram_generator(K)

    best_params = params if params is not None else get_best_params(K, X, Y)
    svr = SVR(kernel=K, C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'], verbose=True)
    return svr.fit(X, Y)
