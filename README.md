# Solving large-scale L1-regularized SVM and cousins: A hybrid column and constraint generation approach

## Getting started
We use the following dependencies
```
Python 2.7.15
Gurobi 6.5.2
```

## Problems solved

We consider a family of regularized linear Support Vectors Machines problem with hinge-loss and convex sparsity-inducing regularization. In particular we study the L1-SVM problem:
```
min \sum_{i=1}^n max (0, 1 - y_i * (x_i^T \beta +\beta_0)) + \lambda \| \beta \|_1,
```
the group-SVM problem:
```
min \sum_{i=1}^n max (0, 1 - y_i * (x_i^T \beta +\beta_0)) + \lambda \sum_{g=1}^G \| \beta_g \|_\inf,
```
and the Slope-SVM problem:
```
min \sum_{i=1}^n max (0, 1 - y_i * (x_i^T \beta +\beta_0)) + \sum_{j=1}^p \lambda_j | \beta_(j) |.
```

## Algorithms

We note n,p the shape of X. For each problem, our method uses First Order Methods to initialize a 
 - Column Generation algorithm when n<<p
 - Constraint Generation algorithm when p<<n
 - Column and Constraint Generation algorithm for both n and p are large.

## Examples

The code in '/example' simulates experiments and compares our best method to a LP solver  -and to our Regularization Path algorithms when p is large- for the following problems:
 - 1/ L1-SVM with n<<p
 - 2/ L1-SVM with p<<n
 - 3/ L1-SVM with n and p large
 - 4/ Group-SVM with n<<p
 - 5/ Slope-SVM with n<<p.




