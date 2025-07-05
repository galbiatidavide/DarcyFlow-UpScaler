# Problem

The first checkpoint requires the development of a numerical upscaling technique to speed up the numerical approximation of single-phase flow in porous media.

The available data consists of recordings made for the [10th SPE Comparative Solution Project, Model 2](https://www.sintef.no/projectweb/geoscale/results/msmfem/spe10/#:~:text=The%20aim%20of%20the%2010th,a%20million%2Dcell%20geological%20model).

The algorithm should output the following information:
- the coarse computational grid, 
- the upscaled permeability tensor.

The algorithm should be uploaded to the group's branch of the main repository following the signature proposed in the checkpoint1.ipynb file.

## Evaluation

The evaluation of the checkpoint will be based on a portion of the available dataset. Your algorithm's performance will be scored out of a maximum of 2 points and the evaluation metric will consider the accuracy of the approximation (1 point) and the efficiency of your implementation (1 point).
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