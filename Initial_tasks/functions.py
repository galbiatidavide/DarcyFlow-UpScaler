import os
import numpy as np
import porepy as pp

import numpy as np


def nearest_spd(A):
    """
    Compute the nearest symmetric positive definite (SPD) matrix to the input matrix A.

    Args:
        A (numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Nearest SPD matrix to A.
    """
    # symmetrize A
    A = make_symmetric(A)

    # Compute the symmetric polar factor of B. Call it H.
    # Clearly H is itself SPD.
    _, S, V = np.linalg.svd(A)
    H = V @ np.diag(S) @ V.T

    # get Ahat in the above formula
    A_spd = make_symmetric((A + H) / 2)

    # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
    k = 0
    while True:
        try:
            assert is_spd(A_spd)
            break
        except AssertionError:
            # Ahat failed the chol test. It must have been just a hair off,
            # due to floating point trash, so it is simplest now just to
            # tweak by adding a tiny multiple of an identity matrix.
            mineig = np.min(np.linalg.eigvals(A_spd))
            A_spd = A_spd + (-mineig * k**2 + np.finfo(float).eps * mineig) * np.eye(
                A.shape[0]
            )

            k += 1

    return A_spd


def make_symmetric(M):
    """
    Make a matrix symmetric by averaging it with its transpose.

    Args:
        M (np.ndarray): The input matrix.

    Returns:
        numpy.ndarray: The symmetric matrix obtained by averaging M with its transpose.
    """
    return 0.5 * (M + M.T)


def is_spd(val):
    """
    Check if a matrix is symmetric positive definite (SPD).

    Args:
        val (np.ndarray): The matrix to be checked.

    Returns:
        bool: True if the matrix is SPD, False otherwise.
    """
    eigs = np.linalg.eigvals(val)
    if np.all(eigs > 0) and np.allclose(val, val.T):
        return True
    else:
        print("The matrix is not positive defined or symmetrix, the eigs are", eigs)
        return False


def coarse_grid(sd, num_part):
    """
    Construct a coarse grid corresponding to the partition of a given grid.

    Args:
        sd (pp.Grid): The original grid to be partitioned.
        num_part (int): The number of partitions to create.

    Returns:
        tuple: A tuple containing the partition structure, the subgrids corresponding to each
            partition, and the coarse grid.

    """
    # Partition the grid
    part = pp.partition.partition_structured(sd, num_part)
    sub_sds, _, _ = pp.partition.partition_grid(sd, part)

    # construct the coarse grid corresponding to the partition
    coarse_dims = pp.partition.determine_coarse_dimensions(num_part, sd.cart_dims)

    x_min, y_min, z_min = sd.nodes[0].min(), sd.nodes[1].min(), sd.nodes[2].min()
    x_max, y_max, z_max = sd.nodes[0].max(), sd.nodes[1].max(), sd.nodes[2].max()

    physical_dims = {
        "xmin": x_min,
        "xmax": x_max,
        "ymin": y_min,
        "ymax": y_max,
        "zmin": z_min,
        "zmax": z_max,
    }

    sd_coarse = pp.CartGrid(coarse_dims, physdims=physical_dims)
    sd_coarse.compute_geometry()

    sd_coarse.full_shape = coarse_dims
    sd_coarse.full_physdims = np.array([x_max - x_min, y_max - y_min])
    sd_coarse.spacing = sd_coarse.full_physdims / coarse_dims

    return part, sub_sds, sd_coarse


def compute_avg_q_grad(sd, p, data, key, bc, bc_val):
    """
    Compute the average cell-centered flux and gradient.

    Args:
        sd (pp.Grid): The grid object.
        p (np.ndarray): The pressure field.
        data (dict): The data dictionary.
        key (str): The key for accessing the data.
        bc (pp.BoundaryCondition): The boundary condition.
        bc_val (np.ndarray): The boundary condition value.

    Returns:
        np.ndarray: The average cell-centered flux.
        np.ndarray: The average cell-centered gradient.
    """
    # Compute the cell-centered flux
    cell_q = compute_P0_flux(sd, p, data, key, bc_val)

    # Compute the cell-centered gradient of the pressure
    cell_grad_p = compute_P0_grad_p(sd, p, key, bc, bc_val)

    # Compute the average cell-centered flux and gradient
    avg_q = np.average(cell_q[: sd.dim], axis=1, weights=sd.cell_volumes)
    avg_grad = np.average(cell_grad_p[: sd.dim], axis=1, weights=sd.cell_volumes)

    return avg_q, avg_grad


def compute_P0_flux(sd, p, data, key, bc_val):
    """
    Compute the P0 flux using the given solution and boundary conditions.

    Args:
        sd (pp.Grid): The grid object.
        p (ndarray): Solution vector.
        data (dict): Dictionary containing discretization matrices and other data.
        key (str): Key to access the discretization matrices in the data dictionary.
        bc_val (np.ndarray): Boundary condition values.

    Returns:
        np.ndarray: The P0 flux reconstruction.

    """
    # Post-process the solution to get the flux
    mat = data[pp.DISCRETIZATION_MATRICES][key]
    q = mat["flux"] @ p + mat["bound_flux"] @ bc_val

    # use the MVEM to get the cell-centered flux
    discr = pp.MVEM(key)
    discr.discretize(sd, data)

    # construct the P0 flux recon
    return discr.project_flux(sd, q, data)


def compute_P0_grad_p(sd, p, key, bc, bc_val):
    """
    Compute the average cell-centered pressure gradient.

    Args:
        sd (pp.Grid): The grid object.
        p (np.ndarray): The pressure field.
        key (str): The key for the discretization method.
        bc (pp.BoundaryCondition): The boundary condition object.
        bc_val (np.ndarray): The boundary condition values.

    Returns:
        np.ndarray: The computed pressure gradient.

    """
    # compute the average cell-centered pressure gradient
    perm_tensor = pp.SecondOrderTensor(np.ones(sd.num_cells))
    parameters = {
        "second_order_tensor": perm_tensor,
        "bc": bc,
        "bc_values": bc_val,
    }
    data = pp.initialize_default_data(sd, {}, key, parameters)

    # Discretize the problem
    discr = pp.Mpfa(key)
    discr.discretize(sd, data)

    return compute_P0_flux(sd, p, data, key, bc_val)


def write_upscaled_perm(file_name, sd, perm, folder_name):
    """
    Write a vtk file for the permeability.

    Args:
        file_name (str): The name of the output file.
        sd (pp.Grid): The grid object.
        perm (list): The perm object.
        folder_name (str): The name of the folder to save the file in.

    Returns:
        None
    """
    # Set up folder structure if not existent.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = os.path.join(folder_name, file_name + ".vtk")

    # Write the header
    lines = ["# vtk DataFile Version 3.0\n"]
    lines.append("vtk output\nASCII\nDATASET UNSTRUCTURED_GRID\n")

    # Write the nodes
    lines.append("POINTS " + str(sd.num_nodes) + " float\n")
    for n in sd.nodes.T:
        lines.append(np.array2string(n)[1:-1] + "\n")

    # Write the cells
    lines.append("CELLS " + str(sd.num_cells) + " " + str(5 * sd.num_cells) + "\n")
    cell_nodes = sd.cell_nodes()
    # Local order of the nodes in the cell
    order = np.array([0, 1, 3, 2])
    for c in np.arange(sd.num_cells):
        nodes = cell_nodes[:, c].nonzero()[0][order]
        lines.append("4 " + np.array2string(nodes)[1:-1] + "\n")

    # Write the cell types
    lines.append("CELL_TYPES " + str(sd.num_cells) + "\n")
    lines += ["9 \n"] * sd.num_cells

    # Write the cell values
    lines.append("CELL_DATA " + str(sd.num_cells) + "\n")
    lines.append("TENSORS k float \n")

    for k in perm:
        lines.append(
            str(k[0])
            + " "
            + str(k[1])
            + " 0 "
            + str(k[2])
            + " "
            + str(k[3])
            + " 0 0 0 0\n"
        )

    # Write the file
    with open(file_name, "w") as f:
        [f.write(l) for l in lines]
