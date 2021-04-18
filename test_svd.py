import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
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
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.show()
# plt.close()
if 0:
    Tgrid, Xgrid = np.meshgrid(t, x)
    plt.contour(Xgrid, Tgrid, np.abs(Y))
    plt.ylim(0, 2)
    plt.show()

U, S, VT = np.linalg.svd(Y, full_matrices=False)
print(U.shape)
print(S.shape)
print(VT.shape)

plt.semilogy(S, '-x')
plt.xlim(0., 10)
plt.show()

plt.plot(x, U[:, 0])
plt.plot(x, U[:, 1])
plt.show()

Y_recon = S[0]*np.outer(U[:, 0], VT[0,:]) + S[1]*np.outer(U[:, 1], VT[1,:])
print(np.linalg.norm(Y_recon-Y))
if 0:
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()
    ax.set_ylim(-2, 2)
    ax.set_xlim(0, len(Y[:, 0]))
    line, = ax.plot(np.arange(len(Y_recon[:, 0])), Y_recon[:, 0], color='k', lw=2)
    # plt.show()
    # exit()
    def animate(i):
        line.set_ydata(Y_recon[:, i])
        # line.set_ydata(Y[:, i])
        return line,
    anim = animation.FuncAnimation(fig, animate,frames=np.arange(100), interval=50, )#blit=True
    plt.show()
    exit()
