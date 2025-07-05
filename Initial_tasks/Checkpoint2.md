# Problem

The second checkpoint consists in finding the optimal location for an extraction well. We consider four injection wells locate at the four corners of the domain.

The problem (in its primal formulation) is formulated as follows: Let $\Omega$ be the domain of interest with boundary $\partial \Omega$ and outward unit normal ${\nu}$. Given 
$k$ the matrix permeability and $f = 0$ the source term, we want to solve the following problem: find $p$ such that
$$
\nabla \cdot (-k \nabla p) = f + b
\quad \text{in } \Omega
$$
with boundary conditions:
$$ \nabla p \cdot \nu = 0 \text{ on } \partial \Omega$$
and 
$$ p(x_e,y_e) = 0 \text{ in } (x_e,y_e) \text{ position of the extraction wells } $$.
The additional source term $b$ encodes a unitary injection rate located in the four corners of the domain.

The goal is to find the optimal location that minimize the pressure on the injecting wells.

To avoid the repetive use of a computationally expensive full-order model, your solution should employ the numerical upscaling technique developed in checkpoint 1 to speed up the numerical approximation of single-phase flow in porous media.

The available data consists of recordings made for the [10th SPE Comparative Solution Project, Model 2](https://www.sintef.no/projectweb/geoscale/results/msmfem/spe10/#:~:text=The%20aim%20of%20the%2010th,a%20million%2Dcell%20geological%20model).

The algorithm should output the following information:
- the estimate of the optmial well location.

The algorithm should be uploaded to the group's branch of the main repository following the signature proposed in the checkpoint2.ipynb file.

## Tasks

This checkpoint requires the completion of three subtasks:
1. Implement a coarse solver based on the upscaling techniques.
2. Verify the correspondence between solutions (in Paraview you can visualize the pressure, the log permeability and the idraulic flux).
3. Develop an optimization routine that takes advantage of both the solvers on the fine and coarse scales.

## Evaluation

The evaluation of the checkpoint will be based on a portion of the available dataset. Your algorithm's performance will be scored out of a maximum of 3 points and the evaluation metric will consider the accuracy of the location estimation (2 points) and the efficiency of your implementation (1 point).
A bonus point will be awarded to the group of students who implement the algorithm with the best balance between efficiency and accuracy.


## Checkpoint organization


### Step 1: problem conceptualization

Create a conceptual model of the problem. 

### Step 2: mathematical formulation

Formulate the problem in a mathematical form.

### Step 3: design of the algorithm

Select a proper strategy to solve the problem.

### Step 4: implementation

Implement your strategy, ensuring that it is fully reproducible by others.

### Step 5: testing phase

Your algorithm will be tested on a test dataset, to verify the effectiveness and efficiency of the procedure.

### Step 6: analysis of the performances

Review your results in view of the modeling and implementation strategies that you have employed.