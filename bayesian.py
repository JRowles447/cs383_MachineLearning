from scipy.special import gamma
import numpy as np
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

class Posterior:
    def __init__(self, limes, cherries, a=2, b=2):
        self.a = a
        self.b = b
        self.limes = limes          # shape: (N,)
        self.cherries = cherries    # scalar int
        self.N = np.shape(self.limes)[0]

    def get_MAP(self):
        """
        compute MAP estimate
        :return: MAP estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        print(self.a)
        print()
        print(self.b)
        print()
        print(self.limes)
        print()
        print(self.cherries)
        print()
        print(self.N)
        print()


        lime_list  = np.zeros(self.N)

        for x in range(self.N):
            # beta prior, beta[a, b]
            # use 1- theta for each value of lime
            theta = (self.cherries + self.a - 1) / (self.cherries + self.limes[x] + self.a + self.b - 2)
            one_minus_theta = 1 - theta
            lime_list[x] = one_minus_theta

        print(lime_list)
        return lime_list

    def get_finite(self):
        """
        compute posterior with finite hypotheses
        :return: estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values

        lime_probs = [0, .25, .50, .75, 1.0]
        hypo_probs = [.1, .2, .4, .2, .1]
        fake = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # return fake
        return np.zeros(self.N)

    def get_infinite(self):
        """
        compute posterior with beta prior
        :return: estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values

        return np.zeros(self.N)

if __name__ == '__main__':
    # Get data
    data = Data()
    limes, cherries = data.get_bayesian_data()

    # Create class instance
    posterior = Posterior(limes=limes, cherries=cherries)

    # PLot the results
    plt.plot(limes, posterior.get_MAP(), label='MAP')
    plt.plot(limes, posterior.get_finite(), label='5 Hypotheses')
    plt.plot(limes, posterior.get_infinite(), label='Bayesian with Beta Prior')
    plt.legend()
    plt.savefig('figures/Q4.png')
