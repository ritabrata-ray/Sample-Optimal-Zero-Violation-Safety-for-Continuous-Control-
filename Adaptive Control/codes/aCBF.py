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


x_0=[0.1999]
theta_est_0=0

def f (x):
    return 1.0*x

def g (x):
    return 1.0

def F (x):
    return x

theta_star=0.5

phi=lambda x: 1-25*x**2

grad_phi=lambda x: -50*x

grad_phi_theta=lambda x:0

def u_nominal (x):
    return -x

#Hyperparameters

theta=0.07
margin=0.0
correction_strength=4

#Simulation Paraneters

Ts=1/4000
U=np.zeros(N)
S=np.zeros(N)
Dot_X=np.zeros(N)
Dot_theta_est=np.zeros(N)
Gamma=5.0
count=0

def tau(x,theta_est):
    return -grad_phi(x)*F(x)

def lamda_cbf(x,theta_est):
    return theta_est-Gamma*grad_phi_theta(x)

def aCBF(x,theta_est):
    un=u_nominal(x)
    u=cp.Variable(1)
    P=np.array([[2.0]])
    q=np.array(-2*un)
    G=np.array([-1*grad_phi(x)*g(x)])
    h=np.array(grad_phi(x)*(f(x)+F(x)*lamda_cbf(x,theta_est)))
    A=np.array([[0.0]])
    b=np.array([0.0])
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(u, P) + q.T @ u),
                 [G @ u <= h,
                  A @ u == b])
    prob.solve()
    if u is None:
        print("solver failed")
    return u.value


def play_dynamics(X,theta_est):
    global count
    #global last_u
    #u=u_nominal(X)
    u=aCBF(X,theta_est)
    if (count < N):
        U[count]=u
        S[count]=X
        Dot_X[count]=f(X)+np.dot(g(X),u)+np.dot(F(X),theta_star)
        #last_u=u
        Dot_theta_est[count]=Gamma*tau(X,theta_est)
        count=count+1
    return (Dot_X[count-1],Dot_theta_est[count-1])

def dynamics_runner(x_0,theta_est_0,t):
    d=t.shape[0]
    X_0=np.copy(x_0)
    T_0=np.copy(theta_est_0)
    X=x_0
    T=theta_est_0
    X=np.expand_dims(X_0,axis=0)
    T=np.expand_dims(T_0,axis=0)
    #print(np.shape(X))
    dt=t[1]-t[0]
    for i in range(d):
        del_X,del_theta_est=play_dynamics(X[i],T[i])
        dX=dt*del_X
        dT=dt*del_theta_est
        #print(dX)
        X_n=X[i]+dX
        T_n=T[i]+dT
        X=np.append(X,[X_n],axis=0)
        T=np.append(T,[T_n],axis=0)
    return (X,T)



X,T=dynamics_runner(x_0,theta_est_0,t)



phi_aCBF=np.zeros(N)
for i in range(N):
    phi_aCBF[i]=phi(S[i])

np.save(os.path.join(data_path,'aCBF'),phi_aCBF)

plt.figure()
plt.plot(t[0:N], phi_aCBF[0:N]); plt.axhline(y = 0, c = 'r');plt.axhline(y=theta,c='g'); plt.axhline(y=theta+margin,c='y'); plt.grid()
plt.xlabel('time in seconds')
plt.ylabel('$phi(X)$')
plt.title('Barrier zeroing function $phi(X)$')
plt.savefig(os.path.join(plot_path,"aCBF_Phi_Plot.png"),format="png", bbox_inches="tight")
plt.show()






