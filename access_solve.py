import numpy as np
import porepy as pp
import scipy.sparse as sps

import sys

main_folder = "./"
spe10_folder = main_folder + "spe10"
sys.path.insert(1, spe10_folder)

from functions_cp2 import *
from spe10 import Spe10

      
def solve_fine(spe10, pos_well, injection_rate=1, well_pressure=0, export_folder=None):
    """
    Compute the averaged gradient and flux for a given subdomain and direction of the pressure
    gradient.

    Args:
        spe10 (object): The object representing the subdomain.
        pos_well (np.ndarray): The position of the production well.
        injection_rate (float, optional): The injection rate of the wells. Defaults to 1.
        well_pressure (float, optional): The pressure at the production well. Defaults to 0.
        export_folder (str, optional): If given, path where to export the results. Defaults to
            None.

    Returns:
        float: The maximum pressure at the injection wells.
    """
    # Extract the grid for simplicity
    sd = spe10.sd
    perm_dict = spe10.perm_as_dict()

    # Permeability
    perm_tensor = pp.SecondOrderTensor(kxx=perm_dict["kxx"])
    # print(perm_tensor)

    # Boundary conditions
    b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]

    # Define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(sd.num_faces)
    bc = pp.BoundaryCondition(sd, b_faces, labels)

    # Collect all parameters in a dictionary
    key = "flow"
    parameters = {"second_order_tensor": perm_tensor, "bc": bc, "bc_values": bc_val}
    data = pp.initialize_default_data(sd, {}, key, parameters)

    # Discretize the problem
    discr = pp.Mpfa(key)
    discr.discretize(sd, data)

    A, b = discr.assemble_matrix_rhs(sd, data)

    # Add the injection wells, all with the same injection rate
    b_wells = np.zeros_like(b)
    index_iwells = [
        0,
        spe10.full_shape[0] - 1,
        spe10.full_shape[0] * spe10.full_shape[1] - spe10.full_shape[0],
        spe10.full_shape[0] * spe10.full_shape[1] - 1,
    ]
    b_wells[index_iwells] = injection_rate

    # Add the production well by using a Lagrange multiplier, first we identify the cell
    ij_well = np.floor((np.asarray(pos_well) / spe10.spacing[:-1])).astype(int)
    # print(ij_well)
    index_pwell = spe10.full_shape[0] * ij_well[1] + ij_well[0]
    vect = np.zeros((sd.num_cells, 1))
    vect[index_pwell] = 1

    # Solve the linear system and compute the pressure by adding the constraint
    A = sps.bmat([[A, vect], [vect.T, None]], format="csc")
    b = np.append(b + b_wells, well_pressure)
    p = sps.linalg.spsolve(A, b)[:-1]

    # extract the discretization matrices build 
    mat_discr = data[pp.DISCRETIZATION_MATRICES][key]

    # reconstruct the flux as post-process
    q_tpfa = mat_discr["flux"] @ p + mat_discr["bound_flux"] @ bc_val

    # to export the flux                                                                                                   
    mvem = pp.MVEM(key)                                                                                      
    mvem.discretize(sd, data)                                                                                                                                                                                     
    # construct the P0 flux reconstruction                                                                                 
    cell_q_mpfa = mvem.project_flux(sd, q_tpfa, data) 
    

    # Export the solution
    # if export_folder is not None:
    #     save = pp.Exporter(sd, "sol", folder_name=export_folder)
    #     save.write_vtu([("p", p), ("log_kxx", np.log10(perm_dict["kxx"])),("q_mpfa", cell_q_mpfa)])

    # Return the maximum pressure at the injection wells
    return np.max(p[index_iwells]), np.argmax(p[index_iwells])



def solve_fine_wrapper(i, j, spe10, x, injection_rate=1, well_pressure=0, export_folder=None):
    result = solve_fine(spe10, x, injection_rate=injection_rate, well_pressure=well_pressure, export_folder=export_folder)
    return (result[0], i, j)






