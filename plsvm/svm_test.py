import numpy as np
from cvxpy import *

# random seed
np.random.seed(1)

# Problem data.
n = 100
i = 20
y = np.random.rand(n)
# A = np.random.rand(n, i)  # normal
A = np.random.rand(i, n).T  # in this order to test random numbers

# Construct the problem.
x = Variable(n)
lmbd = Variable(i)
objective = Minimize(sum_squares(x - y))
constraints = [x == A * lmbd,
               lmbd >= np.zeros(i),
               sum(lmbd) == 1]

prob = Problem(objective, constraints)
result = prob.solve(verbose=True)
print(result)