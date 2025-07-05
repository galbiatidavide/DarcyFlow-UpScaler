import os
import numpy as np
import porepy as pp
import numpy as np
import ray
import scipy.sparse as sps
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
from scipy import stats



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
   
def coarse_grid_dave(sd, num_part):
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
    part = pp.partition.partition_grid(sd, num_part)
    sub_sds, _, _ = pp.partition(sd, part)

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
        

def solve_darcy(selected_layers, pos_well):
    """
    Solves the Darcy flow problem in a porous medium.

    Parameters:
    - selected_layers (list): List of selected layers in the porous medium.
    - pos_well (list): Position of the well in the porous medium.

    Returns:
    - sd (object): The Spe10 object representing the porous medium.
    - perm_dict (dict): Dictionary containing the permeability values of the porous medium.
    - q (array): Array of flux values at each cell in the porous medium.
    - b_faces (array): Array of boundary faces in the porous medium.
    - vect (array): Array representing the well location in the porous medium.
    - M (object): Sparse matrix representing the cell volumes in the porous medium.
    - U (object): Sparse matrix representing the discretized Darcy flow problem.
    - b_upwind (array): Array representing the right-hand side of the Darcy flow problem.

    """
    spe10 = Spe10(selected_layers)
    sd = spe10.sd
    perm_folder = spe10_folder + "/perm/"
    spe10.read_perm(perm_folder)
    perm_dict = spe10.perm_as_dict()

    perm = pp.SecondOrderTensor(kxx=perm_dict["kxx"], kyy=perm_dict["kyy"], kzz=perm_dict["kzz"])      
    injection_rate = 1
    b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = sd.face_centers[:, b_faces]
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(sd.num_faces)
    bc = pp.BoundaryCondition(sd, b_faces, labels)
    parameters = {"second_order_tensor": perm, "bc": bc, "bc_values": bc_val}

    flow_key = "flow"
    flow_data = pp.initialize_default_data(sd, {}, flow_key, parameters)

    mpfa = pp.Mpfa(flow_key)
    mpfa.discretize(sd, flow_data)
    A, b = mpfa.assemble_matrix_rhs(sd, flow_data)
    b_wells = np.zeros_like(b)
    index_iwells = [
        0,
        spe10.full_shape[0] - 1,
        spe10.full_shape[0] * spe10.full_shape[1] - spe10.full_shape[0],
        spe10.full_shape[0] * spe10.full_shape[1] - 1,
        ]
    b_wells[index_iwells] = injection_rate
    ij_well = np.floor( (np.asarray(pos_well) / spe10.spacing[:-1])).astype(int)
    index_pwell = spe10.full_shape[0] * ij_well[1] + ij_well[0]
    vect = np.zeros((sd.num_cells, 1))
    vect[index_pwell] = 1
    A = sps.bmat([[A, vect], [vect.T, None]], format="csc")
    b = np.append(b + b_wells, 0)
    cell_p = sps.linalg.spsolve(A, b)[:-1] # 13200 array
    mat_discr = flow_data[pp.DISCRETIZATION_MATRICES][flow_key]
    q = mat_discr["flux"] @ cell_p + mat_discr["bound_flux"] @ bc_val

    mvem = pp.MVEM(flow_key)                                                                                      
    mvem.discretize(sd, flow_data)                                                                                      
    cell_q = mvem.project_flux(sd, q, flow_data) # (3, 13200) matrix
    save = pp.Exporter(sd, "sol_p", folder_name="transport")
    data_to_export = [("kxx", np.log10(perm_dict["kxx"])), 
                    ("kyy", np.log10(perm_dict["kyy"])), 
                    ("kzz", np.log10(perm_dict["kzz"])),
                    ("cell_p", cell_p),
                    ("cell_q", cell_q)]
    save.write_vtu(data_to_export)

    transport_key = "transport"
    delta_t = 500
    num_steps = 200
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(sd.num_faces)
    bc = pp.BoundaryCondition(sd, b_faces, labels)
    parameters = {"darcy_flux": q, "bc": bc, "bc_values": bc_val}
    transport_data = pp.initialize_default_data(sd, {}, transport_key, parameters)

    upwind = pp.Upwind(transport_key)  #the problem is linear, upwind flux is fine (a bit diffusive)
    upwind.discretize(sd, transport_data)
    U, b_upwind = upwind.assemble_matrix_rhs(sd, transport_data)
    M = sps.csr_matrix(np.diag(sd.cell_volumes), shape=np.shape(U))

    return sd, perm_dict, q, b_faces, vect, M, U, b_upwind


def compute_outflow(initial_conc_pos, sd, perm_dict, q, b_faces, vect, M, U, b_upwind, pos_well, plots=False):
    """
    Computes the outflow from a well in a porous media simulation.

    Parameters:
    - initial_conc_pos (tuple): The initial concentration position as a tuple of (x, y) coordinates.
    - sd: The simulation domain.
    - perm_dict: A dictionary containing the permeability values.
    - q: The flow rate.
    - b_faces: The boundary faces.
    - vect: The vector.
    - M: The mass matrix.
    - U: The upwind matrix.
    - b_upwind: The upwind boundary condition.
    - pos_well: The position of the well.
    - plots (bool): Whether to plot the results or not. Default is False.

    Returns:
    - outflow (list): A list of outflow values at each time step.
    """
    
    kxx = perm_dict["kxx"]
    L = 50
    delta_t = 500
    num_steps = 200
    ini_cond = np.logical_and((np.abs(sd.cell_centers[0,:]-initial_conc_pos[0])<L/2),(np.abs(sd.cell_centers[1,:]-initial_conc_pos[1])<L/2))
    c = np.zeros(sd.num_cells)
    c[ini_cond] = 1

    initial_mass = np.sum(c*sd.cell_volumes)
    save = pp.Exporter(sd, "sol_c", folder_name="transport")
    save.write_vtu([("conc", c)], time_step=0)
    S = M + delta_t * U
    S = sps.bmat([[S, vect], [vect.T, None]], format="csc")

    outflow = []
    lu = sps.linalg.splu(S.tocsc())

    for i in np.arange(num_steps):
        b = M @ c + delta_t * b_upwind
        b=np.append(b,0)
        sol = lu.solve(b)
        c = sol[:-1]
        ll = sol[-1]
        save.write_vtu([("conc", c)], time_step=(i+1)*delta_t)
        outflow.append(ll)
    
    total_outflow = np.sum(outflow)
    final_internal_mass = np.sum(c*sd.cell_volumes)

    if plots==True:
        x = initial_conc_pos[0]
        y = initial_conc_pos[1]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(np.arange(num_steps), outflow)
        plt.title(f'Outflow Data -> {total_outflow/initial_mass*100:.2f}%')
        plt.xlabel(f'Time -> {np.argmax(outflow):.2f}')
        plt.ylabel(f'Outflow -> {np.max(outflow):.2f}')
        
        plt.subplot(1, 3, 2)
        plot_perm(sd, perm_dict, [x, y], pos_well)

        plt.subplot(1, 3, 3)
        kxx_log = np.log10(np.array(perm_dict["kxx"]).flatten())
        plt.hist(kxx_log, bins=40, color='blue', edgecolor='black')
        idx = int((x/6.096)*(y/3.0479999999999996))
        line = np.log10(kxx[13200 - idx])
        plt.axvline(x=line, color='red', linestyle='--', linewidth=2)
        plt.title('Histogram of log10(kxx) Values')
        plt.xlabel('log10(kxx) Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    return outflow
    
 
def generate_neural_network_training_data(sd, kxx, perm_dict, q, b_faces, vect, M, U, b_upwind, initial_conc_pos, pos_well, func="Normal", outputs=False):
    """
    Generate neural network training data based on the given parameters.

    Parameters:
    - kxx (list): List of kxx values.
    - initial_conc_pos (tuple): Tuple containing the initial concentration position (x, y).
    - pos_well (tuple): Tuple containing the position of the well (x, y).
    - outputs (bool): Flag indicating whether to display additional outputs.

    Returns:
    - neural_network_inputs (list): List of neural network input values.
    - neural_network_outputs (list): List of neural network output values.
    """

    x_concentration, y_concentration = np.floor(initial_conc_pos[0] / 6.096), np.floor(initial_conc_pos[1] / 3.0479999999999996)
    x_position_well, y_position_well = np.floor(pos_well[0] / 6.096), np.floor(pos_well[1] / 3.0479999999999996)
    
    neural_network_inputs, neural_network_outputs = [], []
    neural_network_inputs.append(x_concentration)
    neural_network_inputs.append(y_concentration)
    
    x_center, y_center = x_concentration, y_concentration
    radius_big = radius_small = 7
    num_points = 4
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x_points, y_points = np.zeros(num_points), np.zeros(num_points)
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    for i, angle in enumerate(angles):
        if i % 2 == 0:
            x_points[i] = x_center + radius_small * np.cos(angle)
            y_points[i] = y_center + radius_small * np.sin(angle)
        else:
            x_points[i] = x_center + radius_big * np.cos(angle)
            y_points[i] = y_center + radius_big * np.sin(angle)
        
    for x, y in zip(x_points, y_points):
        if 0 < y <= 220 and 0 < x <= 60:
            idx = int(x * y)
            kxx_value = kxx[13200 - idx - 1]
            neural_network_inputs.append(kxx_value)
        elif (x < 0.0 or x > 60) and y > 0:
            idx = int(y)
            kxx_value = kxx[13200 - idx - 1]
            neural_network_inputs.append(kxx_value)
        elif (y < 0.0 or y > 220) and x > 0:
            idx = int(x)
            kxx_value = kxx[13200 - idx - 1]
            neural_network_inputs.append(kxx_value)
        else:
            idx = int(x_center * x_center)
            kxx_value = kxx[13200 - idx - 1]
            neural_network_inputs.append(kxx_value)

    out = compute_outflow(initial_conc_pos, sd, perm_dict, q, b_faces, vect, M, U, b_upwind, pos_well, plots=False)

    if func == "Normal":
        mu, sigma, a = estimate_gaussian_parameters(out)
        neural_network_outputs.append(mu)
        neural_network_outputs.append(sigma)
        neural_network_outputs.append(a)

    elif func == "Exponential":
        mu, a, b = estimate_exponential_parameters(out)
        neural_network_outputs.append(mu)
        neural_network_outputs.append(a)
        neural_network_outputs.append(b)

    if outputs:
        perm_matrix = np.array(kxx).reshape(220,60)
        matrix = np.log10(perm_matrix)
        print(f"x coordinate: {x_concentration}")
        print(f"y coordinate: {y_concentration}")
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.scatter(x_concentration, y_concentration, color='orange', marker='x', s=100, label='concentration')
        plt.scatter(x_position_well, y_position_well, color='yellow', marker='x', s=100, label='well')
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = x_concentration + radius_big * np.cos(theta)
        y_circle = y_concentration + radius_big * np.sin(theta)
        plt.plot(x_circle, y_circle, color='orange', linestyle='--', label='circle')
        for neighbor in neighbors:
            plt.scatter(neighbor[0], neighbor[1], color='red', marker='x')

    return neural_network_inputs, neural_network_outputs   


def estimate_exponential_parameters(out, plots=False):
    y = out
    x = np.linspace(0, 200, 200)
    initial_guess = [50, 20, 5]

    def scaled_exponential(x, mu, a, b):
        return a * np.exp(-(x - mu)) + b

    def objective(params):
        mu, a, b = params
        y_hat = scaled_exponential(x, mu, a, b)
        return np.sum((y - y_hat)**2)

    result = minimize(objective, initial_guess, method='Powell')
    mu_optimal, a_optimal, b_optimal = result.x
    y_hat_optimal = scaled_exponential(x, mu_optimal, a_optimal, b_optimal)

    if plots:
        plt.figure(figsize=(4, 3))
        plt.plot(y, label="Original")
        plt.plot(x, y_hat_optimal, label=f'Scaled Exponential (mu={mu_optimal:.2f}, a={a_optimal:.2f},  b={b_optimal:.2f}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return mu_optimal, a_optimal, b_optimal


def estimate_gaussian_parameters(out, plots=False):
    
    y = out
    x = np.linspace(0, 200, 200)
    initial_guess = [50, 20, 200]

    def normal_distribution(x, mu, sigma, a):
        return a*norm.pdf(x, mu, sigma)

    def objective(params):
        mu, sigma, a = params
        y_hat = normal_distribution(x, mu, sigma, a)
        return np.sum((y - y_hat)**2)
    
    if np.max(y) < 1:
        return 0, 0, 0
    
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    mu_optimal, sigma_optimal, a_optimal = result.x
    y_hat_optimal = normal_distribution(x, mu_optimal, sigma_optimal, a_optimal)

    if plots:
        plt.figure(figsize=(4, 3))
        plt.plot(y, label="Original")
        plt.plot(x, y_hat_optimal, label=f'Normal Distribution (mu={mu_optimal:.2f}, sigma={sigma_optimal:.2f}, a={a_optimal:.2f})')
        plt.legend()
        plt.grid(True)
        plt.show()

    return mu_optimal, sigma_optimal, a_optimal


def compute_training_data(layer):
    spe10 = Spe10(layer)
    sd = spe10.sd
    perm_folder = spe10_folder + "/perm/"
    spe10.read_perm(perm_folder)
    perm_dict = spe10.perm_as_dict()
    kxx = perm_dict["kxx"]

    with open(f'PoreWorld/data/l{layer}.pickle', 'rb') as f: data = pickle.load(f)
    pos_well = data[1][np.argmin(data[2])]

    with open(f'PoreWorld/data/layer_{layer}_list_coordinates.pickle', 'rb') as f: list_initial_conc_pos = pickle.load(f)

    _, _, q, b_faces, vect, M, U, b_upwind = solve_darcy(layer, pos_well)

    list_neural_network_inputs, list_neural_network_outputs = [], []

    idx = 0
    for elem in list_initial_conc_pos:
        neural_network_inputs, neural_network_outputs = generate_neural_network_training_data(sd, kxx, perm_dict, q, b_faces, vect, M, U, b_upwind, elem, pos_well)
        list_neural_network_inputs.append(neural_network_inputs)
        list_neural_network_outputs.append(neural_network_outputs)
        if idx%100 == 0:
            print(f"Progress: {idx/len(list_initial_conc_pos)*100:.2f}%")
        idx += 1

    pickle_filename_input = f"layer_{layer}_list_neural_network_inputs"
    pickle_filename_output = f"layer_{layer}_list_neural_network_outputs"
    with open(f"PoreWorld/data/{pickle_filename_input}.pickle", 'wb') as f:
        pickle.dump(list_neural_network_inputs, f)
        print(f"Result saved to '{pickle_filename_input}'")
    with open(f"PoreWorld/data/{pickle_filename_output}.pickle", 'wb') as f:
        pickle.dump(list_neural_network_outputs, f)
        print(f"Result saved to '{pickle_filename_output}'")

    return


class MLP(nn.Module):
    
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(6, 32).float()
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(32, 16).float()
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(16, 8).float()
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(8, 3).float()
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        
        return x


def normal_distribution(x, mu, sigma, a):
        return a*norm.pdf(x, mu, sigma)
        

def scaled_exponential(x, mu, a, b):
        return a * np.exp(-(x - mu)) + b


