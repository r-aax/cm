"""
System of Linear Equations realization.
"""

import numpy as nm
import scipy.linalg as spla
import random

# ==================================================================================================


class LinearEquation:

    # ----------------------------------------------------------------------------------------------

    def __init__(self, ind, a, b):
        """
        Constructor.
        :param ind: Index of equation.
        :param a: Array of coefficients. a[i] - coefficient for xi member.
        :param b: Right part of value.
        """

        self.Ind = ind
        self.A = a
        self.B = b

# ==================================================================================================


class SystemOfLinearEquations:

    # ----------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Constructor.
        """

        # Create empty system.
        self.N = 0
        self.Equations = []
        self.X = []
        self.Solved = False

    # ----------------------------------------------------------------------------------------------

    def set_ab(self, a, b):
        """
        Create system from given coefficients matrix and right values vector.
        :param a: Matrix of coefficients.
        :param b: Right values vector.
        """

        self.N = len(b)

        # Create equations.
        self.Equations = [LinearEquation(i, a[i], b[i]) for i in range(self.N)]

    # ----------------------------------------------------------------------------------------------

    def set_random(self, n):
        """
        Set random values for coefficients matrix and right values vector.
        :param n: System size.
        """

        mr = lambda m: [random.random() for i in range(m)]
        self.N = n
        self.Equations = [LinearEquation(i, mr(n), random.random()) for i in range(n)]

    # ----------------------------------------------------------------------------------------------

    def print(self):
        """
        Print information.
        """

        print('SystemOfLinearEquations:')

        fformat = '{0:12.8f}'

        # Print each line.
        for i in range(self.N):
            eq = self.Equations[i]
            ai_strs = [fformat.format(aij) for aij in eq.A]
            bi_str = fformat.format(eq.B)
            if (self.N - 1) // 2 == i:
                ch = '='
            else:
                ch = ' '
            ai_str = '{0}) [{1}] [x{2}] {3} [{4}]'.format(eq.Ind, ' '.join(ai_strs), i, ch, bi_str)
            print(ai_str)

        # Print result, if system is solved.
        if self.Solved:
            xi_strs = [fformat.format(x) for x in self.X]
            xi_str = 'X = [{0}]'.format(' '.join(xi_strs))
            print(xi_str)

    # ----------------------------------------------------------------------------------------------

    def solve_scipy(self):
        """
        Colve system using scipy.
        """

        self.X = spla.solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
        self.Solved = True

# ==================================================================================================


if __name__ == '__main__':

    print('Test sle.py module:')
    s = SystemOfLinearEquations()
    s.set_random(3)
    s.solve_scipy()
    s.print()

# ==================================================================================================
