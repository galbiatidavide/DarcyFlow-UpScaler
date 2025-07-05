# 🧪 PoreWorld: An Upscaling Strategy in Porous Media

Scientific Computing Tools for Advanced Mathematical Modelling  
**Academic Year**: 2023–2024  
**Instructor**: Prof. Stefano Pagani  
**Institution**: Politecnico di Milano

---

## 🎯 Project Overview

This repository contains the implementation of **PoreWorld**, a project developed within the course *Scientific Computing Tools for Advanced Mathematical Modelling* at Politecnico di Milano. The main goal of this work is to address the challenge of **upscaling porous media** to simulate fluid flows and transport in geological systems efficiently, without relying on prohibitively expensive fine-grid simulations.

The numerical techniques, parallel computing tools, and surrogate modeling strategies applied here are rooted in real-world applications such as oil reservoir simulation, groundwater hydrology, and contaminant transport modeling.

---

## 📚 Key Technologies and Libraries

The project relies heavily on [**PorePy**](https://github.com/pmgbergen/porepy), an open-source Python library for simulation of flow and transport in deformable fractured media. PorePy enables:

- Grid generation and geometric representation
- Finite volume discretization (MPFA, TPFA)
- Coupled flow and transport problems
- Parallelization and efficient I/O

Other relevant libraries used include:

- `scipy`, `numpy`, `matplotlib`: for numerical computation and visualization
- `pathos.multiprocessing` and `ray`: for parallel processing
- `sklearn.cluster.DBSCAN`: for spatial clustering
- `PyTorch` / `MLPRegressor`: for surrogate modeling (partially successful)

---

## 📝 Abstract

Porous media are materials with voids that may be filled with fluids like air, water, or oil. These structures appear both in nature (e.g., sedimentary rocks) and in technology (e.g., filters, foams).  
Simulating fluid flow through such media at the **microscopic (pore)** scale is computationally infeasible at large domains. Instead, we work at the **macroscopic scale** using the concept of **Representative Elementary Volume (REV)**.  
We upscale local properties (porosity, permeability) into macroscopic tensors using numerical methods. Then, we simulate flow and transport behavior using these upscaled values.

---

## 🧮 Mathematical Formulation

The problem is governed by Darcy's Law in porous media:

$$
q = -\frac{K}{\mu}(\nabla p + \rho g \nabla z)
$$

Where:
- \( q \): volumetric flux
- \( K \): permeability tensor
- \( \mu \): dynamic viscosity
- \( p \): pressure
- \( g \): gravity constant

Porosity is defined via the indicator function over the REV volume:

$$
\phi(x_0, t) = \frac{1}{|\Omega_0|} \int_{\Omega_0} i(x,t)dx
$$

We discretize this problem using a **Multi-Point Flux Approximation (MPFA-0)** scheme.

---

## 🧩 Methodology

The project is split into 3 progressive **Checkpoints**.

### 🔹 Checkpoint 1: Upscaling Permeability Tensor

We estimate a coarse-scale (upscaled) permeability tensor \( K^* \) from fine-grid simulations.  
To ensure symmetry, we add an extra constraint and solve the resulting overdetermined linear system via **least squares**.

Parallelization was implemented via `multiprocessing` for better scalability.

---

### 🔹 Checkpoint 2: Optimal Well Placement via Pressure Minimization

We aim to find the optimal location for a single **extraction well** that minimizes the **maximum pressure** at four **injection wells** located at the corners. The pressure field is governed by a Poisson equation.  
Due to performance constraints on the fine grid, we:
- Perform coarse-grid simulations on 100 and 400-cell grids
- Use DBSCAN clustering to locate “hot zones”
- Refine search only in promising areas

<table>
  <tr>
    <td align="center">
      <strong>Clustering on 100-cells Grid</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/DarcyFlow-UpScaler/main/poreworld_images/clustering_sd100.png" width="300">
    </td>
    <td align="center">
      <strong>Clustering on 400-cells Grid</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/DarcyFlow-UpScaler/main/poreworld_images/clustering_sd400.png" width="300">
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <strong>Overlap of Clusters</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/DarcyFlow-UpScaler/main/poreworld_images/cluster_overlap.png" width="400">
    </td>
  </tr>
</table>

---

## 🔍 Numerical Results

We validated our upscaling approach by comparing original and reconstructed permeability fields.

<table>
  <tr>
    <td align="center">
      <strong>Original Permeability (kxx) - Layer 10</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/DarcyFlow-UpScaler/main/poreworld_images/original_kxx_layer10.png" width="300">
    </td>
    <td align="center">
      <strong>Upscaled Permeability - Layer 10</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/DarcyFlow-UpScaler/main/poreworld_images/upscaled_kxx_layer10.png" width="300">
    </td>
  </tr>
</table>

---

## 📉 Checkpoint 3: Surrogate Model for Contaminant Outflow

We tested two methods to predict the **concentration outflow** profile:

1. ❌ **MLP Neural Network** — fitted parametric distributions (normal or exponential). Poor generalization.
2. ✅ **Spatially Weighted Outflow Averaging** — selected outflows from nearby simulations and interpolated.

Estimated well positions also showed good agreement with ground truth.

<table>
  <tr>
    <td align="center">
      <strong>Estimated vs Real Well Coordinates</strong><br>
      <img src="https://raw.githubusercontent.com/galbiatidavide/DarcyFlow-UpScaler/main/poreworld_images/estimated_vs_real_coordinates.png" width="400">
    </td>
  </tr>
</table>

---

## 🏁 Conclusions

- The upscaling method effectively reduces the computational burden of full-resolution flow simulations.
- Smart zone-based minimization achieves near-optimal well placement efficiently.
- Simpler surrogate modeling outperforms overfit neural approaches in this context.

---

## 📚 References

1. E. Keilegavlen et al. (2017). *Porepy: An open-source simulation tool for flow and transport in deformable fractured rocks.*
2. P. Moritz et al. (2017). *Ray: A distributed framework for emerging AI applications.*
