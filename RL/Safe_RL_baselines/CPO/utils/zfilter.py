import numpy as np
import pdb
# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/

#computes mean, sample variance, and standard deviation of samples
#identify outliers, look at machine learning models accuracy
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        #mean
        self._M = np.zeros(shape)
        #used to calculate variance
        self._S = np.zeros(shape)

    def push(self, x):
        #fix inhomogenous shape error
        if(type(x)==np.float64 or type(x)==int):
            x = np.full(self._M.shape[0],x,dtype=np.float64)
        elif(len(x)==2 and x[1]=={}):
            x = np.asarray(x[0],dtype=object)
        else:
            x = np.asarray(x,dtype=object)
        #pdb.set_trace()
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x[0]
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

#calculate z-score: used to calc probability of a score within std and compare
#scores from different samples with different means/stds
class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)
        self.fix = False

    def __call__(self, x, update=True):
        if(type(x)!=np.ndarray and x==None):
            x=np.zeros(4)
        if update and not self.fix:
            self.rs.push(x)
        if self.demean:
            if(type(x)==np.float64 or type(x)==int):
                x = np.full(4,x,dtype=np.float64)
            #changed to fix dimension issues
            if(len(x)==2 and x[1]=={}):
                x = x[0] - self.rs.mean
            else:
                x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x