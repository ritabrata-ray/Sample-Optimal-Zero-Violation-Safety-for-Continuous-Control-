import numpy as np
import sys
import cvxpy as cp

sys.path.append("..")

from envs.car_yaw_dynamics_4D import car_dynamics



m = 100
I_z = 20  # roughly 1/5 m a^2
a = 1

l=0.2
u=5
C_alpha_f = 10
hat_C_alpha = 7 # clearly between 2 and 50.

def correction_controller(car, u_nominal, theta, eta):

    phi = car.get_CBF_phi()
    grad_phi = car.get_CBF_grad_phi()
    dot_x_minus = car.get_state_derivative_left()

    if (phi > theta):
        #print("Applying nominal control")
        return u_nominal

    #do the decision making above and return nominal. Deviate only if correction control is so necessary.
    #Now we compute the correction controller computation
    # First we write down the basis u_i's
    #print("Applying correction control")
    norm = np.sqrt((1/m)**2 + (a/I_z)**2)
    u_1 = (1/norm)* np.array([1/m, -a/I_z, 0, 0])
    u_2 = (1/norm)* np.array([a/I_z, 1/m, 0, 0])
    u_3 = np.array([0, 0, 1, 0])
    u_4 = np.array([0, 0, 0, 1])
    # Next we compute the SVD matrices U, and V
    V = np.array([[1]])
    U = np.array([u_1, u_2, u_3, u_4])
    Sigma_hat = np.array([[hat_C_alpha * norm],[0],[0],[0]])
    Sigma_hat_pseudo_inverse = np.array([[1/(hat_C_alpha * norm)],[0],[0],[0]])
    hat_g = np.matmul(U, np.matmul(Sigma_hat, np.transpose(V)))
    hat_g_pseudo_inverse = np.matmul(V, np.matmul(  np.transpose(Sigma_hat_pseudo_inverse) ,np.transpose(U)))
    #Now we compute the beta_s
    beta_1 = np.dot(grad_phi, u_1)
    beta_2 = np.dot(grad_phi, u_2)
    beta_3 = np.dot(grad_phi, u_3)
    beta_4 = np.dot(grad_phi, u_4)
    #print("dot phi minus =", np.dot(grad_phi, dot_x_minus))
    #print("norm square of grad_phi =", np.dot(grad_phi, grad_phi))
    #Next we compute alpha
    alpha = (np.dot(grad_phi, dot_x_minus) - eta)/(np.dot(grad_phi, grad_phi))
    # Next we need to solve a convex program to fetch the feasible matrix Gamma:
    Gamma = cp.Variable((4,4))
    constraints = []
    if alpha > 0:
        # Constraint for each i below
        # i=1
        if beta_1 > 0:
            constraints.append( (u_1.T @ (Gamma @ dot_x_minus)) <= ((alpha * beta_1)/u))
        else:
            constraints.append((u_1.T @ (Gamma @ dot_x_minus)) >= ((alpha * beta_1) / u))
        if beta_2 > 0:
            constraints.append( (u_2.T @ (Gamma @ dot_x_minus)) <= ((alpha * beta_2)/u))
        else:
            constraints.append((u_2.T @ (Gamma @ dot_x_minus)) >= ((alpha * beta_2) / u))
        if beta_3 > 0:
            constraints.append( (u_3.T @ (Gamma @ dot_x_minus)) <= ((alpha * beta_3)/u))
        else:
            constraints.append((u_3.T @ (Gamma @ dot_x_minus)) >= ((alpha * beta_3) / u))
        if beta_4 > 0:
            constraints.append( (u_4.T @ (Gamma @ dot_x_minus)) <= ((alpha * beta_4)/u))
        else:
            constraints.append((u_4.T @ (Gamma @ dot_x_minus)) >= ((alpha * beta_4) / u))
    else:
        # Constraint for each i below
        # i=1
        if beta_1 > 0:
            constraints.append((u_1.T @ (Gamma @ dot_x_minus)) <= ((alpha * beta_1) / l))
        else:
            constraints.append((u_1.T @ (Gamma @ dot_x_minus)) >= ((alpha * beta_1) / l))
        if beta_2 > 0:
            constraints.append((u_2.T @ (Gamma @ dot_x_minus)) <= ((alpha * beta_2) / l))
        else:
            constraints.append((u_2.T @ (Gamma @ dot_x_minus)) >= ((alpha * beta_2) / l))
        if beta_3 > 0:
            constraints.append((u_3.T @ (Gamma @ dot_x_minus)) <= ((alpha * beta_3) / l))
        else:
            constraints.append((u_3.T @ (Gamma @ dot_x_minus)) >= ((alpha * beta_3) / l))
        if beta_4 > 0:
            constraints.append((u_4.T @ (Gamma @ dot_x_minus)) <= ((alpha * beta_4) / l))
        else:
            constraints.append((u_4.T @ (Gamma @ dot_x_minus)) >= ((alpha * beta_4) / l))
    # Done with setting up constraints
    prob = cp.Problem(cp.Minimize( cp.norm((Gamma - np.eye(4)),'nuc')), constraints)
    prob.solve()
    #Gamma_matrix = np.zeros((4, 4)) # Get the solution matrix Gamma in numpy
    Gamma_matrix = Gamma.value
    #Finally compute the correction control:
    u_corr = u_nominal - np.matmul(hat_g_pseudo_inverse, np.matmul(Gamma_matrix, dot_x_minus))
    #print("Correction control applied = ", np.clip(u_corr[0], -1000, 1000))
    return np.clip(u_corr[0], -100, 100) # return just the floating point value, no array

"""
car = car_dynamics()
car.reset()
theta = 100
eta = 100
for t in range(1000):
    u = np.random.normal(-10,1)
    u = correction_controller(car, u, theta, eta)
    car.step(u)
    print("CBF = ", car.get_CBF_phi())
    print("State = ", car.get_state())
"""

