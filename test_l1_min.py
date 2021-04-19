import numpy as np
import cvxpy as cvx
n = 20
num_nonzero = int(n*0.2)
num_meas = int(n*0.5)
nonzero_indx = np.random.choice(n, num_nonzero, replace=False)
x_true = np.zeros(n)
x_true[nonzero_indx] = np.random.randn(num_nonzero)
M = np.random.randn(num_meas, len(x_true))
y = M@ x_true

x = cvx.Variable(len(x_true)) #b is dim x  
objective = cvx.Minimize(cvx.norm(x,1)) #L_1 norm objective function
constraints = [M*x == y] #y is dim a and M is dim a by b
prob = cvx.Problem(objective,constraints)
result = prob.solve(verbose=True)
print(x.value)
print(x_true)
exit()
# #then clean up and chop the 1e-12 vals out of the solution
# x = np.array(x.value) #extract array from variable 
# x = np.array([a for b in x for a in b]) #unpack the extra brackets
# x[np.abs(x)<1e-9]=0 #chop small numbers to 0 