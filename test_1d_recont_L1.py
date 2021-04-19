import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import cvxpy as cvx

def creat_signals():
    Nx = 401
    x = np.linspace(-2., 2., Nx)
    dt, Nt = 0.01, 1001
    tend = dt * (Nt-1.)
    t = np.linspace(0., tend, Nt)
    c1,x1,s1,f1 = 1,0.5,0.6,1.3
    c2,x2,s2,f2 = 1.2,-0.5,0.3,4.1

    y1 = c1 * np.exp(-(x-x1)**2/2/s1**2)
    y2 = c2 * np.exp(-(x-x2)**2/2/s2**2)

    Y = np.zeros((Nx, Nt), dtype='d')
    pi = np.pi
    for tt in range(Nt):
        val = y1*np.sin(2*pi*f1*t[tt]) + y2*np.sin(2*pi*f2*t[tt])
        Y[:, tt] = val
    if 0:
        # First set up the figure, the axis, and the plot element we want to animate
        fig, ax = plt.subplots()
        ax.set_ylim(-2, 2)
        ax.set_xlim(0, len(Y[:, 0]))
        line, = ax.plot(np.arange(len(Y[:, 0])), Y[:, 0], color='k', lw=2)
        # plt.show()
        # exit()
        def animate(i):
            line.set_ydata(Y[:, i])
            # line.set_ydata(Y[:, i])
            return line,
        anim = animation.FuncAnimation(fig, animate,frames=np.arange(100), interval=50, )#blit=True
        plt.show()
        exit()
    return Y

data = creat_signals()
num_total_data = data.shape[1]
num_pnts = data.shape[0]
np.random.seed(2)
rand_indx = np.arange(num_total_data)
np.random.shuffle(rand_indx)
data = data[:, rand_indx]


meas_indx = np.random.choice(num_pnts, int(num_pnts*0.05), replace=False)

plt.plot(data[:, 0])
plt.plot(meas_indx, data[meas_indx, 0], 'rx')
plt.show()

train_indx = int(num_total_data*0.5)

train_data = data[meas_indx, 0:train_indx]
target_field_id = train_indx+ 10
y = data[meas_indx, target_field_id]

w = cvx.Variable(train_data.shape[1])
objective = cvx.Minimize(cvx.norm(w,1)) #L_1 norm objective function
constraints = [train_data*w == y] #y is dim a and M is dim a by b
# loss = cvx.sum_squares(train_data @ w - y )/2 + lamda * cvx.norm(w,1)
prob = cvx.Problem(objective,constraints)
result = prob.solve(verbose=True)

# problem = cvx.Problem(cvx.Minimize(loss))
# problem.solve(verbose=True) 
# opt = problem.value
# print('Optimal Objective function value is: {}'.format(opt))
print(w.value)
plt.stem(w.value)
plt.show()

found_basis = np.where(abs(w.value)>0.02)
y_recon = train_data[:, found_basis] @ w.value[found_basis]
plt.plot(data[:, target_field_id], 'r')
plt.plot(meas_indx, y, 'rx')
plt.plot(data[:, found_basis] @ w.value[found_basis], '--g')
plt.plot(meas_indx, y_recon, 'go')
plt.show()