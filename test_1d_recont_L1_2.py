import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc
rc('font',size=16)
rc('font',family='serif')
rc('axes',labelsize=18)
rc('lines', linewidth=2,markersize=10)
def creat_signals():
    Nx = 401
    x = np.linspace(-2., 2., Nx)
    # dt, Nt = 0.01, 1001
    # tend = dt * (Nt-1.)
    tend = 1./1.3
    Nt = 1000
    dt = tend / Nt
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
    return x, Y

x, data = creat_signals()
num_total_data = data.shape[1]
num_pnts = data.shape[0]
np.random.seed(2)
rand_indx = np.arange(num_total_data)
np.random.shuffle(rand_indx)
data = data[:, rand_indx]
mean_data = np.mean(data, axis=1)
std_data = np.std(data, axis=1)
assert len(std_data) == len(mean_data)
for i in range(data.shape[0]):
    data[i, :] -= mean_data[i]
    data[i, :] /= std_data[i]

meas_indx = np.random.choice(num_pnts, int(num_pnts*0.02), replace=False)


train_indx = int(num_total_data*0.5)

train_data = data[meas_indx, 0:train_indx]
target_field_id = train_indx+ 10
y = data[meas_indx, target_field_id]
if 1:
    plt.plot(x, data[:, target_field_id])
    plt.plot(x[meas_indx], y, 'rx')
    plt.show()
    


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

w_caped = np.zeros_like(w.value)
found_basis = np.where(abs(w.value)>0.1)
w_caped[found_basis] = w.value[found_basis]
print(found_basis)
y_exact = (data[:, target_field_id] + mean_data)*std_data
y_recon = (data[:, 0:train_indx] @ w_caped+mean_data)*std_data
# y_recon = train_data[:, found_basis] @ w.value[found_basis]

# plt.plot(data[:, target_field_id], 'r')
# plt.plot(meas_indx, y, 'rx')
# plt.plot(data[:, found_basis] @ w.value[found_basis], '--g')
# plt.plot(meas_indx, y_recon, 'go')
# plt.show()

plt.plot(x[meas_indx], (y+mean_data[meas_indx])*std_data[meas_indx], 'rx', label='meas')
plt.plot(x, y_exact, 'r', label='true sig')
plt.plot(x, y_recon, '--g', label='recov sig')
# plt.plot(meas_indx, (y_recon+mean_data[meas_indx])*std_data[meas_indx], 'go', label='recov meas')
plt.xlabel('x (location)')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()
print(np.linalg.norm(y_exact - y_recon)/np.linalg.norm(y_exact))
