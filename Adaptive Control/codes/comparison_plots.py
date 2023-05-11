import numpy as np
import matplotlib.pyplot as plt
import os

base_path="/Users/ritabrataray/Desktop/Safe control/comparison against safe adaptive control algorithms"

data_path=os.path.join(base_path,"data")

plot_path=os.path.join(base_path,"plots")

aCBF=np.load(os.path.join(data_path,"aCBF.npy"))
RaCBF=np.load(os.path.join(data_path,"RaCBF.npy"))
RaCBF_SMID=np.load(os.path.join(data_path,"RaCBF+SMID.npy"))
CBC=np.load(os.path.join(data_path,"CBC.npy"))
BALSA=np.load(os.path.join(data_path,"BALSA.npy"))
Sign_Flip=np.load(os.path.join(data_path,"Sign-Flip.npy"))

N=Sign_Flip.shape[0]
t=np.arange(N)*(1/4000)

aCBF_t=0
RaCBF_t=0
RaCBF_SMID_t=0
CBC_t=0
BALSA_t=0
Sign_Flip_t=0

def forward_convergence_time(A,N,t):
    x=0
    for i in range(N):
        if(A[i]<0):
            x=t[i]
    return x

aCBF_t=forward_convergence_time(aCBF,N,t)
RaCBF_t=forward_convergence_time(RaCBF,N,t)
RaCBF_SMID_t=forward_convergence_time(RaCBF_SMID,N,t)
CBC_t=forward_convergence_time(CBC,N,t)
BALSA_t=forward_convergence_time(BALSA,N,t)
Sign_Flip_t=forward_convergence_time(Sign_Flip,N,t)

algorithms=['aCBF', 'RaCBF', 'RaCBF+SMID', 'CBC', 'BALSA', 'Our Algorithm']
forward_convergence_times=[aCBF_t, RaCBF_t, RaCBF_SMID_t, CBC_t, BALSA_t, Sign_Flip_t]

plt.bar(algorithms, forward_convergence_times, color=['g','y','c','k','m','b'],width=0.4)

plt.xlabel("Adaptive Safe Control Algorithm")
plt.ylabel("Time after which the system becomes safe")
plt.title("Forward convergence times for different safe control algorithms")
plt.savefig(os.path.join(plot_path,"Bar_Forward_Convergence_Plot.png"),format="png", bbox_inches="tight")
plt.show()


plt.plot(t,Sign_Flip,color='b',label='Our Algorithm')
plt.plot(t,aCBF,color='g',label='aCBF')
plt.plot(t,RaCBF,color='y',label='RaCBF')
plt.plot(t,RaCBF_SMID,color='c',label='RaCBFS')
plt.plot(t,CBC,color='k',label='cbc')
plt.plot(t,BALSA,color='m',label='balsa')

plt.axhline(y = 0, c = 'r',label='safety level',linestyle='--')

#plt.axhline(y = theta, c = 'w', label='safety subset')

plt.xlabel("time")
plt.ylabel("barrier function")
plt.title("Plot for comparison of adaptive safe control algorithms")

plt.legend(loc='best',borderpad=0.4,labelspacing=0.7)
plt.savefig(os.path.join(plot_path,"Adaptive_Safe_Control_Comparison_Plot.png"),format="png", bbox_inches="tight")
plt.show()

