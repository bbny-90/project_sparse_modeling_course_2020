import numpy as np
import cvxpy as cvx

x_true = [0., 3., 0., 0., 0., -1., 0.]
M = np.random.randn(3, len(x_true))
y = M@ x_true

x = cvx.Variable(len(x_true)) #b is dim x  
objective = cvx.Minimize(cvx.norm(x,1)) #L_1 norm objective function
constraints = [M@x == y] #y is dim a and M is dim a by b
prob = cvx.Problem(objective,constraints)
result = prob.solve(verbose=True)
print(x.value)
exit()
# #then clean up and chop the 1e-12 vals out of the solution
# x = np.array(x.value) #extract array from variable 
# x = np.array([a for b in x for a in b]) #unpack the extra brackets
# x[np.abs(x)<1e-9]=0 #chop small numbers to 0 