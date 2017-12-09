import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
from Data import Data


"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because 
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""


class LinearRegression:
    def __init__(self, flag, alpha=1e-2):
        self.flag = flag        # flag can only be either: 'GradientDescent' or 'Analytic'
        self.alpha = alpha      # regularization coefficient
        self.w = np.zeros(2)    # initializing the weights

    def fit(self, X, Y, epochs=10000):
        """
        :param X: input features , shape: (N,2)
        :param Y: target, shape: (N,)
        :return: None

        IMP: Make use of self.w to ensure changes in the weight are reflected globally.
            We will be making use of get_params and set_params to check for correctness
        """
        if self.flag == 'GradientDescent':
            step = 0.001
            for t in range(epochs):
                # WRITE the required CODE HERE to update the parameters using gradient descent
                # do regular gradient descent here from line 1 in slide 27
                res = self.loss_grad(self.w, X, y)
                loss = self.loss(self.w, X, y)

                #converges using step
                new_w = self.w + step * res

                # diverges using alpha
                # new_w = self.w + self.alpha * res


                self.w = new_w


                # make use of self.loss_grad() function
                if t%100 == 0:
                    print("Epoch: {} :: loss: {}".format(t, self.loss(self.w, X, y)))

        elif self.flag == 'Analytic':
            pass

            L2_pseudo = np.dot(X.T, X) + 2 * self.alpha * np.eye(X.shape[1])
            third = np.linalg.inv(L2_pseudo).dot(X.T).dot(y)



            # do closed form from line 2 slide 27
            # Remember: matrix inverse in NOT equal to matrix**(-1)
            # WRITE the required CODE HERE to update the weights using closed form solution
            self.w = third
            print(third)

        else:
            raise ValueError('flag can only be either: ''GradientDescent'' or ''Analytic''')

    def predict(self, X):
        """
        :param X: inputs, shape: (N,2)
        :return: predictions, shape: (N,)

        Return your predictions
        """
        # WRITE the required CODE HERE and return the computed
        y = np.ones(X.shape[0])

        # this is not what IAN said online he said (w.T, X)
        y = np.dot(self.w, X.T)

        return y

    def loss(self, w, X, Y):
        """
        :param W: weights, shape: (2,)
        :param X: input, shape: (N,2)
        :param Y: target, shape: (N,)
        :return: scalar loss value

        Compute the loss loss. (Function will be tested only for gradient descent)
        """
        # WRITE the required CODE HERE and return the computed values
        sum = 0
        row_number = 0
        # get L2 Norm using the formula
        L2_Norm = 0
        for x in w:
            L2_Norm += x**2

        # iterate over the number of rows
        while (row_number < X.shape[0]):
            sum += .5*((np.dot(w.T, X[row_number,]) - Y[row_number])**2 + self.alpha*(L2_Norm))
            row_number+=1
        return sum

    def loss_grad(self, w, X, y):
        """
        :param W: weights, shape: (2,)
        :param X: input, shape: (N,2)
        :param Y: target, shape: (N,)
        :return: vector of shape: (2,) containing gradients for each weight

        Compute the gradient of the loss. (Function will be tested only for gradient descent)
        """

        # WRITE the required CODE HERE and return the computed values
        # using the top gradient
        gradient = 0

        for x in range(X.shape[0]):
            gradient += (y[x]-np.dot(w.T, X[x]))* X[x]
        return gradient

    def get_params(self):
        """
        ********* DO NOT EDIT ************
        :return: the current parameters value
        """
        return self.w

    def set_params(self, w):
        """
        ********* DO NOT EDIT ************
        This function Will be used to set the values of weights externally
        :param w:
        :return: None
        """
        self.w = w
        return 0



if __name__ == '__main__':
    # Get data
    data = Data()
    X, y = data.get_linear_regression_data()

    # Linear regression with gradient descent
    model = LinearRegression(flag='GradientDescent')
    model.fit(X,y)
    y_grad = model.predict(X)
    w_grad = model.get_params()

    # Analytic Linear regression
    model = LinearRegression(flag='Analytic')
    model.fit(X,y)
    y_exact = model.predict(X)
    w_exact = model.get_params()

    # Compare the resultant parameters
    diff = norm(w_exact - w_grad)
    print(diff, w_exact, w_grad)

    # Plot the results
    plt.plot(X[:, 0], y, 'o-', label='true')
    plt.plot(X[:, 0], y_exact, '-', label='pinv')
    plt.plot(X[:, 0], y_grad, '-.', label='grad')
    plt.legend()
    plt.savefig('figures/Q1.png')
    plt.close()
