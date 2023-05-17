import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import control as ct
from matplotlib import animation
from scipy import signal
import math
import cvxpy as cp
import os

base_path="/Users/ritabrataray/Desktop/Safe control/comparison against safe adaptive control algorithms"

data_path=os.path.join(base_path,"data")

plot_path=os.path.join(base_path,"plots")

N=1000
t=np.arange(N)*(1/4000)

#x_0=[0.1999]
x_0=[0.2500] #Now starting at an unsafe position to see recovery.
theta_est_0=0

def f (x):
    return 1.0*x

def g (x):
    return 1.0

def F (x):
    return x


theta_star=0.5

v_max=2.0

phi=lambda x: 1-25*x**2


grad_phi=lambda x: -50*x

grad_phi_theta=lambda x:0

def u_nominal (x):
    return -x

#Hyperparameters

theta=0.07
margin=0.0
D=0.07

#Simulation Parameters

Ts=1/4000
U=np.zeros(N)
S=np.zeros(N)
Dot_X=np.zeros(N)
Dot_theta_est=np.zeros(N)
Gamma=5.0

count=0

#Algorithmic hyperparameters
r_min=0.0
r_max=1.0

def tau(x,theta_est):
    return -grad_phi(x)*F(x)

def lamda_cbf(x,theta_est):
    return theta_est-Gamma*grad_phi_theta(x)

def RaCBF_SMID(x,theta_est,v_max):
    un=u_nominal(x)
    u=cp.Variable(1)
    P=np.array([[2.0]])
    q=np.array(-2*un)
    G=np.array([-1*grad_phi(x)*g(x)])
    h=np.array(grad_phi(x)*(f(x)+F(x)*lamda_cbf(x,theta_est))+phi(x)-((1/(2*Gamma))*(v_max)**2))
    A=np.array([[0.0]])
    b=np.array([0.0])
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(u, P) + q.T @ u),
                 [G @ u <= h,
                  A @ u == b])
    prob.solve()
    if u is None:
        print("solver failed")
    return u.value


def play_dynamics(X, theta_est):
    global count

    # This part is new for SMID.
    # We need to explicitly make sure that|theta_est-theta_star|is indeed within v_max, so we update v_max accordingly.
    global r_min
    global r_max
    global v_max

    X_dot_minus_est = Dot_X[count - 1]

    U_last = U[count - 1]

    # first update r_min and r_max, the U and L limits of theta_star uncertainty using uncertainty bound D.

    if (F(X) > 0):
        r_min = np.maximum(r_min, ((X_dot_minus_est - f(X) - g(X) * U_last - D) / (F(X))))
        r_max = np.minimum(r_max, ((X_dot_minus_est - f(X) - g(X) * U_last + D) / (F(X))))

    else:
        if (F(X) < 0):
            r_max = np.minimum(r_max, ((X_dot_minus_est - f(X) - g(X) * U_last - D) / (F(X))))
            r_min = np.maximum(r_min, ((X_dot_minus_est - f(X) - g(X) * U_last + D) / (F(X))))
        else:
            r_min = r_min
            r_max = r_max  # No need to update if F(X) is zero.

    # Now update v_max from the above information. Only update it after every 10% of the horizon length.

    if (count % 10 == 0):
        v_max = np.maximum(np.abs(theta_est - r_min), np.abs(theta_est - r_max))

    # The SMID related modification part is now over here.

    # global last_u
    # u=u_nominal(X)
    u = RaCBF_SMID(X, theta_est, v_max)
    if (count < N):
        U[count] = u
        S[count] = X
        Dot_X[count] = f(X) + np.dot(g(X), u) + np.dot(F(X), theta_star)
        # last_u=u
        Dot_theta_est[count] = Gamma * tau(X, theta_est)
        count = count + 1
    return (Dot_X[count - 1], Dot_theta_est[count - 1])


def dynamics_runner(x_0, theta_est_0, t):
    d = t.shape[0]
    X_0 = np.copy(x_0)
    T_0 = np.copy(theta_est_0)
    X = x_0
    T = theta_est_0
    X = np.expand_dims(X_0, axis=0)
    T = np.expand_dims(T_0, axis=0)
    # print(np.shape(X))
    dt = t[1] - t[0]
    for i in range(d):
        del_X, del_theta_est = play_dynamics(X[i], T[i])
        dX = dt * del_X
        dT = dt * del_theta_est
        # print(dX)
        X_n = X[i] + dX
        T_n = T[i] + dT
        X = np.append(X, [X_n], axis=0)
        T = np.append(T, [T_n], axis=0)
    return (X, T)

X,T=dynamics_runner(x_0,theta_est_0,t)

phi_RaCBFS=np.zeros(N)
for i in range(N):
    phi_RaCBFS[i]=phi(S[i])

np.save(os.path.join(data_path,'RaCBF+SMID'),phi_RaCBFS)

plt.figure()
plt.plot(t[0:N], phi_RaCBFS[0:N]); plt.axhline(y = 0, c = 'r');plt.axhline(y=theta,c='g'); plt.axhline(y=theta+margin,c='y'); plt.grid()
plt.xlabel('time in seconds')
plt.ylabel('$phi(X)$')
plt.title('Barrier zeroing function $phi(X)$')
plt.savefig(os.path.join(plot_path,"RaCBF+SMID_Phi_Plot.png"),format="png", bbox_inches="tight")
plt.show()




