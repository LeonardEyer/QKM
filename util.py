import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def quantum_dataset():
    # Define constants
    J = 1.0  # Exchange interaction
    hz = 0.2 * J  # External magnetic field
    N = 12  # Number of qubits (for simplicity, assuming 12 qubits)

    # Define Pauli matrices
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    I = qt.qeye(2)

    # Define bond sets for the honeycomb lattice
    # These bonds should be defined based on the honeycomb lattice structure
    X_bonds = [(i, (i + 1) % N) for i in range(0, N, 2)]
    Y_bonds = [(i, (i + 1) % N) for i in range(1, N, 2)]
    Z_bonds = [(i, (i + 1) % N) for i in range(2, N, 2)]

    # Create the Hamiltonian
    def hamiltonian(N, J, hz, X_bonds, Y_bonds, Z_bonds):
        H = 0
        # Adding XX interactions
        for (i, j) in X_bonds:
            H += J * qt.tensor([X if k == i or k == j else I for k in range(N)])
        # Adding YY interactions
        for (i, j) in Y_bonds:
            H += J * qt.tensor([Y if k == i or k == j else I for k in range(N)])
        # Adding ZZ interactions
        for (i, j) in Z_bonds:
            H += J * qt.tensor([Z if k == i or k == j else I for k in range(N)])
        # Adding external magnetic field term
        for i in range(N):
            H += hz * qt.tensor([Z if j == i else I for j in range(N)])
        return H

    H = hamiltonian(N, J, hz, X_bonds, Y_bonds, Z_bonds)

    # Generate a random initial state
    psi0 = qt.tensor([qt.basis(2, np.random.choice([0, 1])) for _ in range(N)])
    # Generate a uniform initial state.
    # psi0 = qt.tensor([qt.basis(2, 0) for _ in range(N)])
    # Normalize the initial state
    psi0 = psi0.unit()

    # Time evolution parameters
    times = np.linspace(0, 10, 51)  # Time range and resolution

    # Observables: Z for each qubit
    observables = [qt.tensor([Z if j == i else I for j in range(N)]) for i in range(N)]

    # Perform time evolution
    result = qt.mesolve(H, psi0, times, [], observables)

    # Calculate total magnetization
    magnetization = np.array([np.sum(expectations) / N for expectations in zip(*result.expect)])

    # Create the dataset
    dataset = np.column_stack((times, magnetization))

    return dataset


def plot_results(training_data, svr_qk, errors, losses):
    """
    Function to plot the given training data, SVR predictions, errors, and losses.

    Parameters:
    - training_data: tuple of (x, y) for the training data.
    - svr_qk: tuple of (x, y) for the SVR QK predictions.
    - errors: dictionary with keys as model names and values as tuples of (x, error).
    - losses: dictionary with keys as model names and values as tuples of (epoch, loss).
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot (a) - Training data and SVR QK predictions
    axs[0].plot(training_data[0], training_data[1], 'c', label='data')
    axs[0].plot(training_data[0], training_data[1], 'go', label='training')
    axs[0].plot(svr_qk[0], svr_qk[1], 'm--', label='SVR QK')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('f(x)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    axs[0].set_title('(a)', loc='left')

    # Plot (b) - Error plots
    colors = ['b', 'c', 'm', 'purple']
    linestyles = ['-', '--', '-.', ':']
    for idx, (model_name, (x, error)) in enumerate(errors.items()):
        axs[1].plot(x, error, color=colors[idx], linestyle=linestyles[idx], label=model_name)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('error')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    axs[1].set_title('(b)', loc='left')

    # Plot (c) - Loss plots
    for idx, (model_name, loss) in enumerate(losses.items()):
        axs[2].plot(loss, color=colors[idx], linestyle=linestyles[idx], label=model_name)
    axs[2].set_xlabel('epoch')
    axs[2].set_ylabel('loss')
    axs[2].set_yscale('log')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)
    axs[2].set_title('(c)', loc='left')

    plt.tight_layout()
    plt.show()


def gram_generator(kernel):
    def k(X, Y):
        result = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                result[i, j] = kernel(X[i][0], Y[j][0])
        return result

    return k


def get_best_params(kernel, x, y):
    param_grid = {
        'C': [1, 10, 100, 1000],
        'gamma': [0.1, 1, 10, 100],
        'epsilon': [0.1, 0.01, 0.001]
    }

    # SVR model with RBF kernel
    svr_classical = SVR(kernel=kernel)

    # GridSearchCV to find the best parameters
    grid_search = GridSearchCV(svr_classical, param_grid, cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(x, y)

    # Best parameters
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)

    return best_params
