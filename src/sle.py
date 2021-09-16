"""
System of Linear Equations realization.
"""

import numpy as np
import scipy.linalg as spla
import random
import time

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
        self.Method = ''
        self.Time = 0.0
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

    def set_random(self, n, a=0.0, b=1.0):
        """
        Set random values for coefficients matrix and right values vector.
        :param n: System size.
        :param a: Start value for generate random numbers.
        :param b: End value for generate random numbers.
        """

        mr = lambda m: [random.uniform(a, b) for i in range(m)]
        self.N = n
        self.Equations = [LinearEquation(i, mr(n), random.uniform(a, b)) for i in range(n)]

    # ----------------------------------------------------------------------------------------------

    def print(self):
        """
        Print information.
        """

        print('SystemOfLinearEquations:')

        ff = '{0:12.8f}'
        ja = lambda a: ' '.join([ff.format(ai) for ai in a])

        # Print system if it is not solved yet.
        if not self.Solved:
            for i in range(self.N):
                eq = self.Equations[i]
                bi_str = ff.format(eq.B)
                if (self.N - 1) // 2 == i:
                    ch = '='
                else:
                    ch = ' '
                ai_str = '{0}) [{1}] [x{2}] {3} [{4}]'.format(eq.Ind, ja(eq.A), i, ch, bi_str)
                print(ai_str)

        # Print result, if system is solved.
        if self.Solved:
            print('   Method = {0}, time = {1:.8f}'.format(self.Method, self.Time))
            print('   X = [{0}]'.format(ja(self.X)))
            diff = self.diff()
            print('   D = [{0}] / {1}'.format(ja(diff), ff.format(np.linalg.norm(diff))))

    # ----------------------------------------------------------------------------------------------

    def collect_a(self):
        """
        Collect A matrix.
        :return: A coefficients matrix.
        """

        # We have to return copy of coefficients.
        return [eq.A[:] for eq in self.Equations]

    # ----------------------------------------------------------------------------------------------

    def collect_b(self):
        """
        Collect B vector of right values.
        :return: B vector of right values.
        """

        return [eq.B for eq in self.Equations]

    # ----------------------------------------------------------------------------------------------

    def solve_scipy(self):
        """
        Solve system using scipy.
        """

        self.Method = 'scipy'
        t = time.time()

        self.X = spla.solve(self.collect_a(), self.collect_b())

        self.Solved = True
        self.Time = time.time() - t

    # ----------------------------------------------------------------------------------------------

    def solve_cramer(self):
        """
        Solve system with Cramer's rule.
        """

        self.Method = 'cramer'
        t = time.time()

        self.X = [0.0] * self.N

        d = np.linalg.det(self.collect_a())
        b = self.collect_b()
        for i in range(self.N):
            a = self.collect_a()
            for j in range(self.N):
                a[j][i] = b[j]
            self.X[i] = np.linalg.det(a) / d

        self.Solved = True
        self.Time = time.time() - t

    # ----------------------------------------------------------------------------------------------

    def div_ith_equation_on_jth_element(self, i, j):
        """
        Normalize i-th equation by division on i-th element.
        :param i: Equation number.
        :param j: Element number.
        """

        eq = self.Equations[i]
        d = eq.A[j]
        for k in range(self.N):
            eq.A[k] /= d
        eq.B /= d

    # ----------------------------------------------------------------------------------------------

    def add_ith_equation_to_jth_equation_with_coeff(self, i, j, q):
        """
        Add i-th equation to j-th equation with k coefficient.
        :param i: First equation.
        :param j: Second equation.
        :param q: Coefficient.
        """

        if i == j:
            raise Exception('add_ith_equation_to_jth_equation_with_coeff: i == j')

        eqi = self.Equations[i]
        eqj = self.Equations[j]
        for k in range(self.N):
            eqj.A[k] += q * eqi.A[k]
        eqj.B += q * eqi.B

    # ----------------------------------------------------------------------------------------------

    def gauss_step_forward(self, i):
        """
        Step of Gauss method with equation index i.
        :param i: Equation index.
        """

        self.div_ith_equation_on_jth_element(i, i)
        for j in range(i + 1, self.N):
            q = -self.Equations[j].A[i]
            self.add_ith_equation_to_jth_equation_with_coeff(i, j, q)

    # ----------------------------------------------------------------------------------------------

    def gauss_steps_forward(self):
        """
        Steps forward of Gauss method.
        """

        for i in range(self.N):
            self.gauss_step_forward(i)

    # ----------------------------------------------------------------------------------------------

    def gauss_step_back(self, i):
        """
        Step of Gauss method with equation index i.
        :param i: Equation index.
        """

        for j in range(i - 1, -1, -1):
            q = -self.Equations[j].A[i]
            self.add_ith_equation_to_jth_equation_with_coeff(i, j, q)

    # ----------------------------------------------------------------------------------------------

    def gauss_steps_back(self):
        """
        Steps back of Gauss method.
        """

        for i in range(self.N - 1, -1, -1):
            self.gauss_step_back(i)

    # ----------------------------------------------------------------------------------------------

    def solve_gauss(self):
        """
        Solve system of equations with Gauss' method.
        """

        self.Method = 'gauss'
        t = time.time()

        # Steps forward and back.
        self.gauss_steps_forward()
        self.gauss_steps_back()

        # Copy values from right vector to X.
        self.X = [self.Equations[i].B for i in range(self.N)]

        self.Solved = True
        self.Time = time.time() - t

    # ----------------------------------------------------------------------------------------------

    def diff(self):
        """
        Calculate diff vector.
        :return: Diff vector.
        """

        npa = np.array(self.collect_a())
        npx = self.X
        npb = np.array(self.collect_b())

        return npb - np.dot(npa, npx)

# ==================================================================================================


if __name__ == '__main__':

    print('Test sle.py module:')
    s = SystemOfLinearEquations()
    s.set_random(10)
    s.print()
    s.solve_gauss()
    s.print()

# ==================================================================================================
