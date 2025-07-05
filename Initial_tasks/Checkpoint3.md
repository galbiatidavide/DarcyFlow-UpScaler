# Problem

The third checkpoint consists in constructing a surrogate model for the outflow concentration of a quantity under investigation at the optimal extraction well (identified in checkpoint 2) given the initial location of the concentration. As in checkpoint 2, we consider four injection wells locate at the four corners of the domain.

The forward problem consists in a transport problem with advective field resulting from the Darcy problem of Checkpoint 2.

The goal of this checkpoint is to build an efficient and accurate surrogate model that approximate the outflow concentration profile at the optimal location that minimize the pressure on the injecting wells.

The available permeability data consists of recordings made for the [10th SPE Comparative Solution Project, Model 2](https://www.sintef.no/projectweb/geoscale/results/msmfem/spe10/#:~:text=The%20aim%20of%20the%2010th,a%20million%2Dcell%20geological%20model).

The algorithm should output the following information:
- the outflow profile given the location of the initial concentration and the selected layer.

The algorithm should be uploaded to the group's branch of the main repository following the signature proposed in the checkpoint3.ipynb file.

## Tasks

This checkpoint requires the completion of two subtasks:
1. Identify the optimal location of the extraction well for the layers [51, 71, 42, 20, 12, 2, 82].
2. train a surrogate models that takes advantage of the provided high-fidelity solver.

## Evaluation

The evaluation of the checkpoint will be based on the layers [51, 71, 42, 20, 12, 2, 82]. Your algorithm's performance will be scored out of a maximum of 2 points and the evaluation metric will consider the accuracy of the outflow.
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