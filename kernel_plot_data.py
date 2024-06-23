import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from kernels import K_classical, K_L_prod, K_L_cheb, K_L_tower, K_layered, K_phi_RX, K_L_tower_HA

kernels = {
    'XYZ': K_layered,
    'RBF': K_classical,
    'L-prod': K_L_prod,
    'L-tower': K_L_tower,
    'L-cheb': K_L_cheb,
    'phi_RX': K_phi_RX,
    'L-tower_HA': K_L_tower_HA
}

# Function to compute the kernel values for a given kernel
def compute_kernel_values(k, xs):
    return [[k(x, y) for x in xs] for y in tqdm(xs, leave=False)]


# Function to parallelize the computation
def parallel_compute(kernels, xs):
    plot_data = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        for kernel_row in tqdm(kernels, desc="Kernel Rows"):
            args = [(k, xs) for k in kernel_row]
            row_results = list(executor.map(lambda x: compute_kernel_values(*x), args))
            plot_data.append(row_results)
    return plot_data


def gen_data(label, extent=2 * np.pi):
    print(f"Generating data for {label}")
    k = kernels[label]

    kernels_instances = [
        [k(2, n_layers=1), k(4, n_layers=1), k(8, n_layers=1)],
        [k(2, n_layers=2), k(4, n_layers=2), k(8, n_layers=2)],
        [k(2, n_layers=3), k(4, n_layers=3), k(8, n_layers=3)]
    ]

    xs = np.linspace(0, extent, 100)

    plot_data = np.array(parallel_compute(kernels_instances, xs))

    # Save the plot data
    np.save(f"data/[{label}]_kernel_plot_data.npy", plot_data)


def plot_kernel_data(plot_data, label, save=False, extent=2 * np.pi):

    # check if nan values are present
    if np.isnan(plot_data).any():
        print("NaN values present in the data")

    fig, axs = plt.subplots(3, 3, figsize=(8, 8), layout='compressed', sharex=True, sharey=True)
    #fig.suptitle(f'{label} kernel', fontsize=16)

    # Compute the vmin and vmax for consistent color mapping across all plots
    vmin = np.min([np.min(row) for row in plot_data])
    vmax = np.max([np.max(row) for row in plot_data])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps['inferno']
    cmap.set_bad(color='white')

    im = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plotting the data
    for i in range(3):
        for j in range(3):
            axs[i, j].imshow(plot_data[i][j], extent=[0, extent, 0, extent], norm=norm, cmap=cmap, origin='lower')

    # Add extra labels to the top and left/right
    layer_labels = ['L=1 Layer', 'L=2 Layers', 'L=3 Layers']
    qubit_labels = ['N=2 Qubits', 'N=4 Qubits', 'N=8 Qubits']

    # Setting the labels on outer plots
    for i in range(3):
        # Setting the labels on the top
        axs[0, i].set_title(qubit_labels[i], fontsize=10)

        axs[2, i].set_xlabel('x')
        axs[i, 0].set_ylabel(f'{layer_labels[i]}\ny')

        extent_label = []
        if extent == 2 * np.pi:
            extent_label = ["0", r"$\pi$", r"$2\pi$"]
        else :
            extent_label = ["0", f"{extent/2}", f"{extent}"]

        axs[2, i].set_xticks([0, extent / 2, extent], extent_label)
        axs[i, 0].set_yticks([0, extent / 2, extent], extent_label)

    fig.colorbar(im, ax=axs.ravel().tolist())
    if save:
        plt.savefig(f"data/{label}_kernel_plot.png", transparent=True)
    else:
        plt.show()


if __name__ == '__main__':

    label = 'L-cheb'
    #gen_data(label, extent=1)

    plot_data = np.load(f"data/[{label}]_kernel_plot_data.npy", allow_pickle=True)

    #k = kernels[label]
    #k = k(2, n_layers=2)

    #data = compute_kernel_values(k, np.linspace(-1, 1, 100))
    #plt.imshow(data, extent=[0, 2 * np.pi, 0, 2 * np.pi])
    #plt.show()
    plot_kernel_data(plot_data, label, extent=1, save=True)
