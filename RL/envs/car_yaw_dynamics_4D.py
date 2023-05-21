import numpy as np

""" The car takes a right turn of 90 degrees without moving much laterally, and without moving too fast."""


"""
The car has the following 4-dimensional non-linear Yaw dynamics: 
(SEE PAGES 18-20 OF State_Space_Control pdf)
    d(V_y)/dt = (-c_0/(m * V)) * V_y + (-c_1/(m * V) - V) * r + 0 * psi + 0 * y +(C_alpha_f/m) * delta
    d(r)/dt = (-c_1/(I_z * V)) * V_y + (-c_2/(I_z * V) - V) * r + 0 * psi + 0 * y +(a * (C_alpha_f/I_z)) * delta
    d(psi)/dt = 0 * V_y + 1 * r + 0 * psi + 0 * y + 0 * delta
    d(y)/dt =  cos ( psi ) * V_y + 0 * r + V_x * sin( psi ) + 0 * y + 0 * delta
    
delta is the turning angle: the only control variable
V is the velocity which has components, V_x, V_y along x- and y- axis satisfying: V = \sqrt{ V_x^2 + V_y^2 }, the magnitude V is constant
c_0 and c_1 are unknown tyre friction force and tyre reverse momentum type components and the third term in dot V_y is probably centrifugal force
c_1 and c_2 are similar unknown tyre friction resistive parameters, I_z is the known moment of Inertia along the vertical axis, C_alpha_f is unknown tyre rubber friction force coefficient
a is the known distance of front wheels from the center of mass of the car, m is the known mass of the car
"""

#VALUES OF CONSTANT PARAMETERS: BOTH KNOWN AND UNKNOWN


V = 5
c_0 = 70
c_1 = 40
c_2 = 180
m = 100
I_z = 20  # roughly 1/5 m a^2
a = 1
C_alpha_f = 10


"""
The goal is to turn the car from North to East i.e., psi goes from 0 to pi/2, by only applying control variable: delta in the range [-1,1].
The state space is continuous 4-D. Policy network should take this as an input.
The action space is continuous 1-D. Policy network will return 2 outputs the mean and variance of delta. 
Perform the Gaussian reparameterization trick here:, i.e., take Gaussian variable z ~ N (0,1), and play delta = sqrt (variance) * z + mean
reward = -1 * time * time_penalty + achievement_weight * 1 / ((new_psi - pi/2)^2) ____ time_penalty=1, achievement_weight = 2.5 * 0.1
cost = w_r*(r-r_ref)^2 + w_y*(y)^2 + w_lateral_velocity * (V_y - V_y_ref =0?)^2
CBF_function phi = C - cost
DONE = TRUE iff |psi - pi/2| < 0.01
time-horizon = 1000
if not DONE and time-horizon == 1000, info = FAIL
initialize: V_y = 0, max = 10, min =-10; psi = 0 , max = pi, min = -pi/2, r = 0, y = 0
"""

class car_dynamics:
    def __init__(self, V_y_init = 0, psi_init = 0, r_init = 0, y_init = 0):


        # Discrete time step for running dynamics
        self.tau = 0.02
        self.horizon = 1000
        self.time = 0

        self.state_dim = 4
        self.action_dim = 1

        self.V_y = V_y_init
        self.psi = psi_init
        self.r = r_init
        self.y = y_init
        self.V_x = V

        self.V_y_MAX = 7  # Realistic upper bounds on magnitude
        self.r_MAX = 350  # Realistic upper bounds on magnitude
        self.V_MAX = 11   # Realistic upper bounds on magnitude

        # Parameters needed to compute cost and CBF, CBF_grad
        self.continuous_time_horizon = self.tau * self.horizon
        self.r_ref = 100 * ((np.pi/2)/(self.continuous_time_horizon)) # ideally want to make the turn in 1% of the total time horizon
        self.y_ref = 0 # don't like lateral movement of the car, V_y should go from 0 to V, so V_y should not be on a different scale
        self.V_y_ref = V/2
        self.w_r = 4  #may tune them later
        self.w_y = 0  #may tune them later, was 1 earlier
        self.w_lateral_velocity = 0.001 # Only matters if the lateral velocity exceeds the right scales, may tune it later!

        self.state_trajectory = np.zeros((self.horizon + 1, self.state_dim))
        self.state_trajectory[0][0] = self.V_y
        self.state_trajectory[0][1] = self.r
        self.state_trajectory[0][2] = self.psi
        self.state_trajectory[0][3] = self.y

    def get_V(self):
        return np.clip(np.sqrt((self.V_x**2) + (self.V_y**2)),-1*self.V_MAX,self.V_MAX)

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_time_axis(self):
        return (self.tau) * np.arange(self.horizon)

    def reset(self):
        self.V_y = 0
        self.psi = 0
        self.r = 0
        self.y = 0
        self.time = 0
        self.V_x = V

    def get_f(self):
        V = self.get_V()
        f=np.zeros(self.state_dim)
        f[0] = (-(c_0)/(m * V)) * self.V_y + (-(c_1)/(m * V) -V) * self.r
        f[1] = (-(c_1)/(I_z * V)) * self.V_y + (-(c_2)/(I_z * V)) * self.r
        f[2] = self.r
        f[3] = np.cos(self.psi) * self.V_y + self.V_x * np.sin(self.psi) * self.psi
        return f

    def get_g(self):
        g = np.zeros((self.state_dim, self.action_dim))
        g[0][0] = ((C_alpha_f)/(m))
        g[1][0] = ((a*(C_alpha_f))/I_z)
        g[2][0] = 0
        g[3][0] = 0 # actually 0; trying 10 to check controllability in safety
        return g

    # Function to run the dynamics
    def step(self, action):
        # We have the following dynamics equation:
        # \dot x1=f1(x1,x2)+lambda_1*u1
        # \dot x2=f2(x1,x2)+lambda_2*u2
        # Here we take f1(x1,x2)=0.1*(x1-x2)^2, f2(x1,x2)=0.007*|x2|, lambda_1=2, lambda_2=-1
        DONE = False
        info = "SUCC"
        action = np.array([action])
        state_prev = self.state_trajectory[self.time]
        f = self.get_f()
        g = self.get_g()
        state_derivative = f + np.matmul(g,action)
        state_delta = state_derivative * self.tau
        state = state_prev + state_delta
        self.time = self.time + 1
        self.state_trajectory[self.time] = state
        if (self.time >= 1000):
            info = "FAIL"
            DONE = True
        self.V_y = np.clip(state[0],-1*self.V_y_MAX, self.V_y_MAX)
        self.r = np.clip(state[1],-1*self.r_MAX, self.r_MAX)
        self.psi = self.reduce_angles_modulo_2_pi(state[2])
        self.y = state[3]  #No such upper bound on y, but gets a high cost if this is far from zero
        reward = -4 + (0.25) * (1 / (self.psi - ((np.pi) / 2)) ** 2 + 1e-4)
        if (np.abs((self.psi-(np.pi)/2)) < (np.pi/36)):
            DONE = True
            reward = 7000
        if (reward > 7000):
            reward = 7000

        cost = self.w_r * (self.r-self.r_ref)**2 + self.w_y * (self.y-self.y_ref)**2 + self.w_lateral_velocity * (self.V_y - self.V_y_ref)**2
        if (cost > 1e+2):
            cost = 1e+2
        cost /= 1000

        return reward, cost, DONE, info

    def reduce_angles_modulo_2_pi(self, theta):
        if ((theta >= 0) and (theta < 2*np.pi)):
            return theta
        else:
            rounds = (theta//(2*np.pi))
            theta -= rounds*(2*np.pi)
            if theta < 0:
                theta += 2*np.pi
            return theta

    def get_CBF_phi(self):
        self.C = 200
        return self.C - (self.w_r * (self.r-self.r_ref)**2 + self.w_y * (self.y-self.y_ref)**2 + self.w_lateral_velocity * (self.V_y - self.V_y_ref)**2)

    def get_CBF_grad_phi(self):
        grad = np.zeros(self.state_dim)
        grad[0] = (-2) * ((self.w_lateral_velocity) * (self.V_y - self.V_y_ref))
        grad[1] = (-2) * ((self.w_r) * (self.r - self.r_ref))
        grad[2] = 0
        grad[3] = (-2) * ((self.w_y) * (self.y - self.y_ref))
        return grad

    def get_state(self):
        state = np.zeros(self.state_dim)
        state[0] = self.V_y
        state[1] = self.r
        state[2] = self.psi
        state[3] = self.y
        return state.astype(np.float32)

    def get_state_derivative_left(self):
        x=self.state_trajectory[self.time]
        x0=self.state_trajectory[self.time -1]
        xc=x-x0
        xd=(1/self.tau)*xc
        return xd

    def get_state_trajectory(self):
        return self.state_trajectory

"""
#Trial run with a random normal policy
car = car_dynamics()
car.reset()
rewards=[]
costs=[]
succ = 0
time = car.get_time_axis()
# A trajectory
for t in range(1000):
    action = np.random.normal(0,1)*0.1
    reward, cost, DONE, info = car.step(action)
    rewards.append(reward)
    costs.append(cost)
    if (DONE == True):
        if (info == "SUCC"):
            succ = 1
        print("{} at step # {}".format(info, t))
        break

trajectory = car.get_state_trajectory()
print("Rewards:")
print(rewards)
print("Costs:")
print(costs)
"""


