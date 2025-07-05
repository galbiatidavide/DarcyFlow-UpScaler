import os
import numpy as np
import matplotlib.pyplot as plt

def main(layer_num, plot=False):
    nx, ny, nz = 60, 220, 85
    n = nx*ny
    N = n*nz

    perm = np.loadtxt("spe_perm.dat").ravel()
    phi = np.loadtxt("spe_phi.dat").ravel()

    k = np.array([perm[:N], perm[N:2*N], perm[2*N:]]).T

    folder_perm = "perm/"
    folder_phi = "phi/"

    if not os.path.exists(folder_perm):
        os.makedirs(folder_perm)

    if not os.path.exists(folder_phi):
        os.makedirs(folder_phi)

    # loop layer by layer
    for layer in np.atleast_1d(layer_num):
        # extract the permeability and porosity
        k_layer = k[layer*n:(layer+1)*n, :]
        phi_layer = phi[layer*n:(layer+1)*n]

        # save the permeability and porosity on file
        file_name = str(layer) + ".tar.gz"

        np.savetxt(folder_perm + file_name, k_layer, delimiter=",")
        np.savetxt(folder_phi + file_name, phi_layer, delimiter=",")

        if plot:
            k_layer = np.log10(k_layer[:, 0].reshape((ny, nx)).T)
            plt.imshow(k_layer)
            plt.show()

            plt.imshow(phi_layer.reshape((ny, nx)).T)
            plt.show()


if __name__ == "__main__":

    layer_id = np.arange(85)
    main(layer_id, plot=False)
