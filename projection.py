""" 
This code defines methods to solve the MD probeleme. 
The original code can be found in this Git repo: "https://github.com/walidk/AcceleratedMirrorDescent". 
The methodes implemented in this code are those defined in the article: "https://papers.nips.cc/paper_files/paper/2015/file/f60bb6bb4c96d4df93c51bd69dcc15a0-Paper.pdf".

"""

import math
import random
import numpy as np

# ######################################################################################################################
# Simplex projections
# ######################################################################################################################
class SimplexPotentialProjection(object):
    def __init__(self, potential, inversePotential, strong_convexity_const, precision = 1e-10):
        self.inversePotential = inversePotential
        self.gradPsi = np.vectorize(potential)
        self.gradPsiInverse = np.vectorize(inversePotential)
        self.precision = precision
        self.strong_convexity_const = strong_convexity_const
    def __str__(self):
        return 'Simplex potential projection'
    def project(self, x, g):
        d = len(x)
        y = float(max(self.gradPsiInverse(x) - g))
        nuU = self.inversePotential(1) - y
        nuL = self.inversePotential(1/d) - y
        def xBar(nu):
            return np.maximum(0, self.gradPsi(self.gradPsiInverse(x) - g + nu))
        while(sum(xBar(nuU) - xBar(nuL)) > self.precision):
            nuM = (nuU+nuL)/2
            if(sum(xBar(nuM)) > 1):
                nuU = nuM
            else:
                nuL = nuM
        return xBar(nuU)

class SimplexProjectionEuclidean(SimplexPotentialProjection):
    def __init__(self, precision = 1e-10):
        def potential(x):
            return x
        def inversePotential(x):
            return x
        super().__init__(potential, inversePotential, 1, precision)
    def __str__(self):
        return 'SimplexEuclidean'

class SimplexProjectionPNorm(SimplexPotentialProjection):
    def __init__(self, p = 2, precision = 1e-10):
        if(p <= 1 or p > 2):
            raise Exception("In p norm potentials, p must be in (1, 2]")
        def potential(x):
            return np.sign(x)*np.power((p-1)*np.abs(x), 1/(p-1))
        def inversePotential(x):
            return np.sign(x)*np.power(np.abs(x), p-1)/(p-1)
        super().__init__(potential, inversePotential, 1, precision)
    def __str__(self):
        return 'Simplex{}Norm'.format(self.p)


class SimplexProjectionExp(SimplexPotentialProjection):
    def __init__(self, dimension, epsilon = 0, precision = 1e-10):
        self.epsilon = epsilon
        def potential(x):
            return np.exp(x) - epsilon
        def inversePotential(x):
            return np.log(x+epsilon)
        super().__init__(potential, inversePotential, 1./(1+dimension*epsilon), precision)
    def __str__(self):
        return 'SimplexExp{}'.format(self.epsilon)

class SimplexProjectionExpSort(SimplexProjectionExp):
    def __init__(self, dimension, epsilon = 0):
        self.epsilon = epsilon
        super().__init__(dimension, epsilon)
    def __str__(self):
        return 'SimplexExpSort{}'.format(self.epsilon)

    def project(self, x, g):
        """Computes the Bregman projection, with exponential potential, of a vector x given a gradient vector g, using a
        sorting method. The complexity of this method is O(d log d), where d is the size of x.
        Takes as input
        - the current iterate x
        - the gradient vector (scaled by the step size) g
        """
        epsilon = self.epsilon
        d = len(x)
        y = (x+epsilon)*np.exp(-g)
        yy = sorted(y)
        S = sum(yy)
        j = 0
        while((1+epsilon*(d-j))*yy[j]/S - epsilon <= 0):
            S -= yy[j]
            j += 1
        return np.maximum(0, -epsilon+(1+epsilon*(d-j))*y/S)

# ######################################################################################################################
# Unconstrained
# ######################################################################################################################
class PotentialProjection(object):
    def __init__(self, potential, inversePotential, strong_convexity_const):
        self.inversePotential = inversePotential
        self.gradPsi = np.vectorize(potential)
        self.gradPsiInverse = np.vectorize(inversePotential)
        self.strong_convexity_const = strong_convexity_const
    def project(self, x, g):
        return self.gradPsi(self.gradPsiInverse(x) - g)
    def __str__(self):
        return 'Potential'

class PotentialProjectionExp(PotentialProjection):
    def __init__(self, dimension, epsilon = 0):
        self.epsilon = epsilon
        def potential(x):
            return np.exp(x) - epsilon
        def inversePotential(x):
            return np.log(x+epsilon)
        super().__init__(potential, inversePotential, 1/(1+dimension*epsilon))
    def __str__(self):
        return 'PotentialExp{}'.format(self.epsilon)

class NoProjection(PotentialProjection):
    def __init__(self):
        def potential(x):
            return x
        def inversePotential(x):
            return x
        super().__init__(potential, inversePotential, 1)
    # override the project method
    def project(self, x, g):
        return (x - g)
    def __str__(self):
        return 'Identity'


