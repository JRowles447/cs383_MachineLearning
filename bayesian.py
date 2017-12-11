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

        lime_probs = [0, .25, .50, .75, 1.0]
        hypo_probs = [.1, .2, .4, .2, .1]

        lime_list = np.zeros(self.N)

        for lime in range(self.limes.shape[0]):
            # iterate over all the hypotheses
            sum_lime = 0
            sum_cherry = 0
            for i in range(5):
                local_sum_lime = lime_probs[i]*hypo_probs[i]
                local_sum_cherry = (1-lime_probs[i])*hypo_probs[i]
                # iterate over all the samples
                product = 1
                for j in range(self.N):
                    print(lime)
                    if (j < self.limes[lime]):
                        product = product * lime_probs[i]
                    else:
                        product = product * (1- lime_probs[i])
                sum_lime += local_sum_lime * product
                sum_cherry += local_sum_cherry * product
            final_sum = (sum_lime/(sum_lime+sum_cherry))
            print("sum of limes is: " + str(sum_lime))
            print("sum of cherries is: " + str(sum_cherry))
            print("final sum: " + str(final_sum))
            print("cherry sum: " + str((sum_cherry)/(sum_lime + sum_cherry)))
            lime_list[lime] = final_sum

            print(lime_list[lime])
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
        return fake

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
