import pennylane as qml
import pennylane.numpy as np


# Classical kernel, k(x, y) = exp(-(x-y)**2/(2*sigma**2))
def K_classical(sigma):
    def k(x, y):
        return np.exp(-(x - y) ** 2 / (2 * sigma ** 2))

    return k


def make_kernel(psi, N, diff_method="best", qml_backend='default.qubit',**backend_options):
    """
    Make a quantum kernel from a quantum feature map psi with N qubits.
    """
    dev = qml.device(qml_backend, wires=N, **backend_options)

    def k(x, y):
        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
        def kernel_circuit(x, y):
            psi(y)
            qml.adjoint(psi)(x)
            return qml.probs(wires=range(N))

        return kernel_circuit(x, y)[0]

    return k


# Quantum kernel with a single Rx gate on every qubit as a feature map
def K_simple_RX(N, **kwargs):
    def U(x):
        for n in range(N):
            qml.RX(x, wires=n)

    return make_kernel(U, N, **kwargs)


def K_phi_RX(N, **kwargs):
    def phi(x, q):
        return q * x / 2

    def U(x):
        for n in range(N):
            qml.RX(phi(x, n), wires=n)

    return make_kernel(U, N, **kwargs)


def K_layered(N, n_layers, **kwargs):
    def phi(x, q):
        return q * x / 2

    def U(x):

        def CNOT_entangler():
            # CNOT entangler (alternate for each layer)
            if layer % 2 == 0:
                for wire in range(0, N - 1, 2):
                    qml.CNOT(wires=[wire, wire + 1])
            else:
                for wire in range(1, N - 1, 2):
                    qml.CNOT(wires=[wire, wire + 1])

                qml.CNOT(wires=[N - 1, 0])

        def U_k(pauli="X"):
            for n in range(N):
                if pauli == "X":
                    qml.RX(phi(x, n), wires=n)
                elif pauli == "Y":
                    qml.RY(phi(x, n), wires=n)
                elif pauli == "Z":
                    qml.RZ(phi(x, n), wires=n)

        for i, layer in enumerate(range(n_layers)):

            if layer % 3 == 0:
                U_k("X")
            elif layer % 3 == 1:
                U_k("Y")
            elif layer % 3 == 2:
                U_k("Z")

            # If last layer, no entangler
            if layer == n_layers - 1:
                continue

            CNOT_entangler()

    return make_kernel(U, N, **kwargs)


def K_L_prod(N, n_layers, HEA=qml.StronglyEntanglingLayers, **kwargs):
    """
    A single layered product feature map with $|\\psi(x)\\rangle = U(x)V|0\\rangle$
    where $V$ is a HEA of depth five with randomized parameters which are set throughout training
    and $U(x)=\\prod_{j}^{N}(R^{j}_X(x))$
    """
    shape = HEA.shape(n_layers=5, n_wires=N)

    weights = [np.random.random(size=shape) for _ in range(n_layers)]

    def V(params):
        HEA(params, wires=range(N))

    def U(x):
        for j in range(N):
            qml.RX(x, wires=j)

    def psi(x):
        for weight in weights:
            V(weight)
            U(x)

    return make_kernel(psi, N, **kwargs)


def K_L_tower(N, n_layers, **kwargs):
    """
    A single layered product feature map with $|\\psi(x)\\rangle = U(x)V|0\\rangle$
    where $V$ is a HEA of depth five with randomized parameters which are set throughout training
    and $U(x)=\\prod_{j}^{N}(R^{j}_X(j * x))$
    """
    shape = qml.StronglyEntanglingLayers.shape(n_layers=5, n_wires=N)
    weights = [np.random.random(size=shape) for _ in range(n_layers)]

    def V(params):
        qml.StronglyEntanglingLayers(params, wires=range(N))

    def U(x):
        for j in range(N):
            qml.RX(j * x, wires=j)

    def psi(x):
        for weight in weights:
            V(weight)
            U(x)

    return make_kernel(psi, N, **kwargs)


def K_L_cheb(N, n_layers, **kwargs):
    """
        A single layered product feature map with $|\\psi(x)\\rangle = U(x)V|0\\rangle$
        where $V$ is a HEA of depth five with randomized parameters which are set throughout training
        and $U(x)=\\prod_{j}^{N}(R^{j}_X(j * \arccos(x)))$
        """
    shape = qml.StronglyEntanglingLayers.shape(n_layers=5, n_wires=N)
    weights = [np.random.random(size=shape) for _ in range(n_layers)]

    def V(params):
        qml.StronglyEntanglingLayers(params, wires=range(N))

    def U(x):
        for j in range(N):
            qml.RX(j * np.arccos(x), wires=j)

    def psi(x):
        for weight in weights:
            V(weight)
            U(x)

    return make_kernel(psi, N, **kwargs)


def K_L_tower_HA(N, n_layers, HEA=qml.BasicEntanglerLayers, **kwargs):
    """
        A single layered product feature map with $|\\psi(x)\\rangle = U(x)V|0\\rangle$
        where $V$ is a HEA of depth five with randomized parameters which are set throughout training
        and $U(x)=\\prod_{j}^{N}(R^{j}_X(j * x))$
        """
    shape = HEA.shape(n_layers=5, n_wires=N)
    weights = [np.random.random(size=shape) for _ in range(n_layers)]

    def V(params):
        HEA(params, wires=range(N))

    def U(x):
        for j in range(N):
            qml.RX(j * x, wires=j)

    def psi(x):
        for weight in weights:
            V(weight)
            U(x)

    return make_kernel(psi, N, **kwargs)
