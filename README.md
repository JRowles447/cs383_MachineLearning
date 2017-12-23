# cs383_MachineLearning

Homework assignment \#6 for Artificial Intelligence course. Includes programs for linear and logistic regression, k-means, and bayesian learning (also includes decision tree, but not yet implemented). All programs produce figures that can be located in `/figures`. Linear regression runs on set data, logistic regression operates on random data, k-means runs on random points with randomly initialized centroids.

## Running the Programs
### Running Linear Regression
Run Linear Regression with the following command:    
`>python linearRegression.py`   
Program approximates weight function through multiple iterations of gradient descent. Calculates the closed form solution through the utilization of the Moore-Penrose pseudoinverse. Output includes the difference between the approximate and analytic calculations and both weight vectors.


### Running Logistic Regression
Run Logistic Regression with the following command:    
`>python logisticRegression.py`   

Sample Graph:
![Alt text](images/logistic_regression.png)
### Running kmean
Run kmeans with the following command:    
`>python kmeans.py`   
Program iteratively assigns vertices to clusters (through euclidean distance), reassigns the centroid locations as the center of all the points in cluster assignment.

Sample Graph:

### Running Bayesian Learning
