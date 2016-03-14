from __future__ import absolute_import
from __future__ import print_function
from scipy.linalg import cho_factor, cho_solve
import numpy as np
import copy
import ctypes
from itertools import product
from six.moves import range
from six.moves import zip

class Kernel(object):
    def __init__(self, pars, **kwargs):
        self.ndim = kwargs.get("ndim", 1)
        self.pars = np.atleast_1d(pars)
        assert len(self.pars) == self.ndim, "Number of parameters must equal number of dimensions"
        self.computed = False
        self.changepoints = None

    def __add__(self,k2):
        if isinstance(k2,Kernel):
            return Sum(self,k2)
        else:
            pars = [np.fabs(k2) for i in range(self.ndim)]
            return Sum(self,ConstantKernel(pars,ndim=self.ndim))
                    
    def __radd__(self,k2):
        return self._add__(k2)
        
    def __mul__(self,k2):
        if isinstance(k2,Kernel):            
            return Product(self,k2)
        else:
            pars = [np.fabs(k2) for i in range(self.ndim)]
            return Product(self,ConstantKernel(pars,ndim=self.ndim))
        
    def __rmul__(self,k2):
        return self.__mul__(k2)
        
    def get_covar(self):
        if not self.computed:
            raise Exception("Must compute covariance matrix first")
        return self.covar

    def precompute(self,x1,x2):
        """Needed for changepoint kernels.
        
        For a changepoint kernel we need to work out the indices
        corresponding to the changepoints.
        
        Do nothing for normal kernels
        """
        pass
        
    def compute(self,x,errs):
        self.computed = False
        
        x = np.atleast_2d(x)
        assert (x.shape[0] == self.ndim) or (x.shape[0] == 1), "1st dimension of x array must either match dimensions of kernel or be 1"
        assert len(errs) == x.shape[1], "Length of error array must match 2nd dimension of x array"
        
        # precompute for changepoint kernels
        self.precompute(x,x)
        
        if x.shape[0] == 1:
            x = np.vstack([x for i in range(self.ndim)])
        num_points = len(errs)
        # diagonal part of covariance matrix (white noise terms)
        self.covar = errs*errs*np.eye(num_points)

        for i in range(self.ndim):
            # use numpy broadcasting to make 2x2 array of time differences between points
            deltaT = x[i,:]-x[i,:][:,np.newaxis]
            self.covar += self._evaluate(deltaT,i)

        self.factor, self.flag = cho_factor(self.covar)
        self.logdet = 2*np.sum(np.log(np.diag(self.factor)))
        self.computed = True
        
    def get_matrix(self,x1,x2):
        #check x1
        x1 = np.atleast_2d(x1)
        
        
        assert (x1.shape[0] == self.ndim) or (x1.shape[0] == 1), "1st dimension of x1 array must either match dimensions of kernel or be 1"
        if x1.shape[0] == 1:
            x1 = np.vstack([x1 for i in range(self.ndim)])
            
        #check x2
        x2 = np.atleast_2d(x2)
        assert (x2.shape[0] == self.ndim) or (x2.shape[0] == 1), "1st dimension of x1 array must either match dimensions of kernel or be 1"
        if x2.shape[0] == 1:
            x2 = np.vstack([x2 for i in range(self.ndim)])

        # precompute for changepoint kernels
        self.precompute(x1,x2)
        
        matrix = np.zeros((x1.shape[1],x2.shape[1]))
        for i in range(self.ndim):
            X1, X2 = np.meshgrid(x1[i,:],x2[i,:],indexing='ij')
            deltaT = X1-X2
            matrix += self._evaluate(deltaT,i)
        return matrix
        

class DrasticChangepointKernel(Kernel):
    """Implementation of drastic changepoint kernel from Osborne et al.
    The assumption is that the change in hyperparameters is so large that 
    observations before the changepoint are completely uninformative about
    points after the changepoint"""
    def __init__(self,kernels,changepoints):
        for kernel in kernels:
            assert kernel.ndim == 1, "Only 1D Changepoint Kernels are supported"
        assert len(changepoints)+1 == len(kernels), "Must have one fewer changepoints than kernels"
        self.kernels = copy.deepcopy(kernels)
        self.changepoints = copy.deepcopy(changepoints)
        self.ndim = 1
        self.computed = False
        
    def compute(self,x,errs):
        # common stuff for all points
        self.computed = False
        assert x.ndim == 1, "Only 1D kernels are supported"
        assert len(errs) == len(x), "Length of error array must match 2nd dimension of x array"

        num_points = len(errs)
        # diagonal part of covariance matrix (white noise terms)
        self.covar = errs*errs*np.eye(num_points)

        # now the stuff for changepoints
        '''assume x is sorted'''
        breaks = [np.argmax(x>cp) for cp in self.changepoints if (x.min() <= cp <= x.max())]
        # split x arr up into bits
        xarrs  = np.split(x,breaks)
        # start indices and end indices for gram matrices
        startInds = np.insert(breaks,0,0)
        endInds   = np.append(breaks,num_points)
        
    
        for xarr, kernel, startInd, endInd in zip(xarrs, self.kernels, startInds, endInds):
            # create deltaT array
            deltaT = x[startInd:endInd]-x[startInd:endInd][:,np.newaxis]
            covar = kernel._evaluate(deltaT,0) 
            # insert gram matrix in correct place 
            self.covar[startInd:endInd,startInd:endInd] += covar            

        self.factor, self.flag = cho_factor(self.covar)
        self.logdet = 2*np.sum(np.log(np.diag(self.factor)))
        self.computed = True
    
    def get_matrix(self,x1,x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        assert x1.shape[0] == 1, "Only 1D kernels are supported"
        assert x2.shape[0] == 1, "Only 1D kernels are supported"
        x1 = x1[0]
        x2 = x2[0]
        matrix = np.zeros((len(x1),len(x2)))
        
        breaks1 = [np.argmax(x1>cp) for cp in self.changepoints if (x1.min() <= cp <= x1.max())]
        breaks2 = [np.argmax(x2>cp) for cp in self.changepoints if (x2.min() <= cp <= x2.max())]
        
        # split x arr up into bits
        x1arrs  = np.split(x1,breaks1)                
        x2arrs  = np.split(x2,breaks2)  
        
        # start indices and end indices for gram matrices
        startInds1 = np.insert(breaks1,0,0)
        endInds1   = np.append(breaks1,len(x1))
        startInds2 = np.insert(breaks2,0,0)
        endInds2   = np.append(breaks2,len(x2))
        
        for x1arr, x2arr, kernel, s1, e1, s2, e2 in \
            zip(x1arrs, x2arrs, self.kernels, startInds1, endInds1, startInds2, endInds2):

            X1M, X2M = np.meshgrid(x1[s1:e1],x2[s2:e2],indexing='ij')
            deltaT = X1M-X2M
            matrix[s1:e1,s2:e2] += kernel._evaluate(deltaT,0)
        return matrix
                                          
class Sum(Kernel):
    def __init__(self,k1,k2):
        assert k1.ndim == k2.ndim, "Dimension Mismatch"
        self._k1 = k1
        self._k2 = k2
        self.computed = False
        self.ndim = k1.ndim

    def precompute(self,x1, x2):
        self._k1.precompute(x1, x2)
        self._k2.precompute(x1, x2)
                
    def _evaluate(self,deltaT,idim):
        return self._k1._evaluate(deltaT,idim) + self._k2._evaluate(deltaT,idim)
        
class Product(Kernel):
    def __init__(self,k1,k2):
        assert k1.ndim == k2.ndim, "Dimension Mismatch"
        self._k1 = k1
        self._k2 = k2
        self.ndim = k1.ndim
        self.computed = False
    
    def precompute(self, x1, x2):
        self._k1.precompute(x1, x2)
        self._k2.precompute(x1, x2)
            
    def _evaluate(self,deltaT,idim):
        return self._k1._evaluate(deltaT,idim) * self._k2._evaluate(deltaT, idim)
                
class ConstantKernel(Kernel):
    def _evaluate(self,deltaT,idim):
        tau = self.pars[0]
        return tau*np.ones_like(deltaT)
  
class OutputScaleChangePointKernel(Kernel):
    """Kernel with drastic changes in output scale
    
    See Osborne et al or sect 4.4 of Garnett et al (2010) for details"""
    def __init__(self, pars, changepoints, **kwargs):
        assert pars.ndim == 1, "Only 1D Changepoint Kernels are supported"
        assert len(changepoints)+1 == len(pars), "Must have one fewer changepoints than pars"
        self.changepoints = np.array(changepoints)
        self.cpIndices    = None
        self.pars         = np.array(pars)
        self.ndim = 1
        self.computed = False
        
    def precompute(self, x1, x2):
        """Calculate indices of changepoints"""
        self.cpIndices = []
        # locations of changepoints in x1
        self.cpIndices.append( [np.argmax(x1>cp) for cp in self.changepoints if (x1.min() <= cp <= x1.max())] )
        self.cpIndices[0].insert(0,0)
        self.cpIndices[0].append(x1.shape[1])
        # locations of changepoints in x2
        self.cpIndices.append(  [np.argmax(x2>cp) for cp in self.changepoints if (x2.min() <= cp <= x2.max())] )
        self.cpIndices[1].insert(0,0)
        self.cpIndices[1].append(x2.shape[1])
        
    def _evaluate(self,deltaT,idim):
        # like a constant kernel, but with tau dependent on deltaT and changepoints
        
        # convert CPs in each axis into a list of slice obects 
        start_of_slices = self.cpIndices[0][:-1]
        end_of_slices   = self.cpIndices[0][1:]
        slices0 = [slice(s,e) for s,e in zip(start_of_slices,end_of_slices)]

        start_of_slices = self.cpIndices[1][:-1]
        end_of_slices   = self.cpIndices[1][1:]
        slices1 = [slice(s,e) for s,e in zip(start_of_slices,end_of_slices)]

        slice_combinations = product(slices0,slices1)
        par_combinations = product(self.pars,self.pars)

        out = np.zeros_like(deltaT)
        for par_pair, slice_pair in zip(par_combinations, slice_combinations):               
            #tau = np.sqrt(par_pair[0]*par_pair[1]) if par_pair[0]==par_pair[1] else 0
            tau = np.sqrt(par_pair[0]*par_pair[1])
            out[slice_pair[0],slice_pair[1]] = tau
        return out
                
class ExpKernel(Kernel):
    def _evaluate(self,deltaT,idim):
        return np.exp(-np.fabs(deltaT/np.sqrt(self.pars[idim])))
        
class ExpSquaredKernel(Kernel): 
    def _evaluate(self,deltaT,idim):          
        return np.exp(-0.5*(deltaT**2/self.pars[idim]))

class Matern32Kernel(Kernel):
    def _evaluate(self,deltaT,idim):
        return (1+np.sqrt(3*deltaT**2/self.pars[idim])) * \
            np.exp(-np.sqrt(3*deltaT**2/self.pars[idim]))
        
class GaussianProcess(object):
    def __init__(self,kernel):
        assert isinstance(kernel,Kernel)
        self.kernel = kernel
        
    def compute(self, x, errs):
        self.kernel.compute(x,errs)
        # save computed values for later use
        self._x = np.atleast_2d(x)
        
    def lnlikelihood(self,residuals):
        assert self.kernel.computed, "Must compute kernel before calculating lnlikelihood"
        return -0.5*(np.dot(residuals,cho_solve((self.kernel.factor,self.kernel.flag), residuals)) + self.kernel.logdet + len(residuals)*np.log(2.0*np.pi))
        
    def predict(self, y, xp):
        '''Compute the conditional predictive distribution of the model.
        :param y: ``(nsamples,)``
            The observations to condition the model on
        :param xp: ``(ntest,)``
            The coordinates where the predictive sample should be computed
        :returns mu: ``(ntest,)``
            The mean of the predictive distribution
        :returns cov: ``(ntest,ntest)``
            The predictive covariance
        '''
        assert self.kernel.computed, "Must compute kernel before predicting"
        xp = np.atleast_2d(xp)
        assert (xp.shape[0] == self.kernel.ndim) or (xp.shape[0] == 1), "1st dimension of xp array must either match dimensions of kernel or be 1"
        assert len(y) == self._x.shape[1], "Observations have different length to computed positions"

        K1 = self.kernel.get_matrix(xp,self._x)
        K2 = self.kernel.get_matrix(xp,xp)
        mu = np.dot(K1,cho_solve((self.kernel.factor,self.kernel.flag),y))
        cov = K2 - np.dot(K1,cho_solve((self.kernel.factor,self.kernel.flag),K1.T))
        return (mu, cov)
       
    def sample_conditional(self, y, xp, size=100):
        '''
        Draw samples from the predictive conditional distribution.

        :param y: ``(nsamples, )``
            The observations to condition the model on.

        :param xp: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        :param size: (optional)
            The number of samples to draw.

        :returns samples: ``(size, ntest)``
            A list of predictions at coordinates given by ``t``.     
        '''
        mu,cov = self.predict(y,xp)
        return np.random.multivariate_normal(mu,cov,size)

    def sample(self,xp,size=100):
        '''
        Draw samples from the prior distribution
        
        :param xp: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the prior distribution should be computed
        :param size: (optional)
            The number of samples to draw.

        :returns samples: ``(size, ntest)``
            A list of predictions at coordinates given by ``xp``.     
        '''
        assert self.kernel.computed
        xp = np.atleast_2d(xp)
        assert (xp.shape[0] == self.kernel.ndim) or (xp.shape[0] == 1), "1st dimension of xp array must either match dimensions of kernel or be 1"

        cov = self.kernel.get_matrix(xp,self._x)
        mu = np.zeros_like(xp.ravel())
        return np.random.multivariate_normal(mu,cov,size)
            

def test_cps():
    from astropy.convolution import Gaussian1DKernel, convolve
    x = np.linspace(0,20,500)
    xalt = np.linspace(0,20,350)
    e = 0.1*np.ones_like(x)
    
    before, after = 0.01,0.5
    #before cp1
    mask = x<10
    k = before*ExpSquaredKernel(0.3)
    gp = GaussianProcess(k)
    gp.compute(x[mask],e[mask])
    y1 = gp.sample(x[mask],size=1)[0]

    
    #after cp
    mask = x>=10
    k = after*ExpSquaredKernel(0.3)
    gp = GaussianProcess(k)
    gp.compute(x[mask],e[mask])
    y2 = gp.sample(x[mask],size=1)[0]
    
    # scale so continuous
    y2 += y1[-1]-y2[0]
    y = np.concatenate((y1,y2))
    
    # create filter
    g = Gaussian1DKernel(stddev=2)
    # Convolve data
    y = convolve(y, g, boundary='extend')


    k1 = ExpSquaredKernel(0.3)
    pars = np.array([before,after])
    changePoints = np.array([10])
    k2 = OutputScaleChangePointKernel(pars,changePoints)
    kernel = k2*k1
    gp = GaussianProcess(kernel)
    gp.compute(x,e)
    
    mat1 = kernel.get_matrix(x,xalt)
    #plt.imshow(mat)
    #plt.show()

    xt = np.linspace(0,20,1000)
    samples = gp.sample_conditional(y,xt)
    mu = np.mean(samples,axis=0)
    fmu, _ = gp.predict(y, xt)
    std = np.std(samples,axis=0)
    
    plt.plot(x,y,'k-')
    plt.plot(xt,fmu,'r--')
    plt.fill_between(xt,mu-std,mu+std,color='r',alpha=0.5)

    k1 = before*ExpSquaredKernel(0.3)
    k2 = after*ExpSquaredKernel(0.3)
    kernel = DrasticChangepointKernel([k1,k2],changePoints)
    gp = GaussianProcess(kernel)
    gp.compute(x,e)
    xt = np.linspace(0,20,1000)
    samples = gp.sample_conditional(y,xt)
    mu = np.mean(samples,axis=0)
    std = np.std(samples,axis=0)
    plt.fill_between(xt,mu-std,mu+std,color='g',alpha=0.2)
    
    mat2 = kernel.get_matrix(x,xalt)
    fig, ax = plt.subplots(nrows=1,ncols=3)
    plt.subplots_adjust(wspace=0.0)
    ax[0].imshow( mat1 )
    ax[1].imshow( mat2 )
    ax[2].imshow( mat1-mat2)
    plt.show()
    
if __name__ == "__main__":
    import george
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    """
    k1 = 2.0*george.kernels.ExpSquaredKernel(3.0) + 1.0*george.kernels.Matern32Kernel(2.0)
    k2 = 2.0*ExpSquaredKernel(3.0) + 1.0*Matern32Kernel(2.0)
    gp1 = george.GP(k1)
    x = np.linspace(0,30,100)
    e = 0.01*np.ones_like(x)

    fig, ax = plt.subplots(nrows=1,ncols=3)
    plt.subplots_adjust(wspace=0.0)
    ax[0].imshow( gp1.get_matrix(x) )
    ax[1].imshow( k2.get_matrix(x,x) )
    X = k2.get_matrix(x,x)/gp1.get_matrix(x)
    print(X.mean(), X.std())
    ax[2].imshow(X)
    plt.show()
    """
    test_cps()
    