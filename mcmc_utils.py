import numpy as np
import scipy.signal as signal
import scipy.optimize as opt
import scipy.stats as stats
import triangle
from progress import ProgressBar
import scipy.integrate as intg
import warnings
from matplotlib import pyplot as plt
TINY = -np.inf

class Prior(object):
    '''a class to represent a prior on a parameter, which makes calculating 
    prior log-probability easier.

    Priors can be of four types: gauss, gaussPos, uniform and log_uniform

    gauss is a Gaussian distribution, and is useful for parameters with
    existing constraints in the literature
    gaussPos is like gauss but enforces positivity
    Gaussian priors are initialised as Prior('gauss',mean,stdDev)

    uniform is a uniform prior, initialised like Prior('uniform',low_limit,high_limit)
    uniform priors are useful because they are 'uninformative'

    log_uniform priors have constant probability in log-space. They are the uninformative prior
    for 'scale-factors', such as error bars (look up Jeffries prior for more info)'''
    def __init__(self,type,p1,p2):
        assert type in ['gauss','gaussPos','uniform','log_uniform']
        self.type = type
        self.p1   = p1
        self.p2   = p2
        if type == 'log_uniform' and self.p1 < 1.0e-30:
            warnings.warn('lower limit on log_uniform prior rescaled from %f to 1.0e-30' % self.p1)
            self.p1 = 1.0e-30
        if type == 'log_uniform':
            self.normalise = 1.0
            self.normalise = np.fabs(intg.quad(self.ln_prob,self.p1,self.p2)[0])

    def ln_prob(self,val):
        if self.type == 'gauss':	
            return np.log( stats.norm(scale=self.p2,loc=self.p1).pdf(val) )
        elif self.type == 'gaussPos':
            if val <= 0.0:
                return TINY
            else:
                return np.log( stats.norm(scale=self.p2,loc=self.p1).pdf(val) )
        elif self.type == 'uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0/np.abs(self.p1-self.p2))
            else:	
                return TINY
        elif self.type == 'log_uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0 / self.normalise / val)
            else:	
                return TINY
		 	
class Param(object):
	'''A Param needs a starting value, a current value, and a prior'''
	def __init__(self,startVal,prior):
		self.startVal = startVal
		self.prior    = prior
		self.currVal  = startVal
		
def fracWithin(pdf,val):
	return pdf[pdf>=val].sum()

def thumbPlot(chain,labels,**kwargs):
    fig = triangle.corner(chain,labels=labels,**kwargs)
    return fig

def scatterWalkers(pos0,percentScatter):
    warnings.warn('scatterWalkers decprecated: use emcee.utils.sample_ball instead')
    nwalkers = pos0.shape[0]
    npars    = pos0.shape[1]
    scatter = np.array([np.random.normal(size=npars) for i in xrange(nwalkers)])
    return pos0 + percentScatter*pos0*scatter/100.0

def run_burnin(sampler,startPos,nSteps,storechain=False):
    iStep = 0
    bar = ProgressBar()
    for pos, prob, state in sampler.sample(startPos,iterations=nSteps,storechain=storechain):
        bar.render(int(100*iStep/nSteps),'running Burn In')
        iStep += 1
    return pos, prob, state
    
def run_mcmc_save(sampler,startPos,nSteps,rState,file):
    '''runs and MCMC chain with emcee, and saves steps to a file'''
    #open chain save file
    if file:
        f = open(file,"w")
        f.close()
    iStep = 0
    bar = ProgressBar()
    for pos, prob, state in sampler.sample(startPos,iterations=nSteps,rstate0=rState,storechain=True):
        if file:
            f = open(file,"a")
        bar.render(int(100*iStep/nSteps),'running MCMC')
        iStep += 1
        for k in range(pos.shape[0]):
            # loop over all walkers and append to file
            thisPos = pos[k]
            thisProb = prob[k]
            if file:
                f.write("{0:4d} {1:s} {2:f}\n".format(k," ".join(map(str,thisPos)),thisProb ))
        if file:        
            f.close()
    return sampler
    
def flatchain(chain,npars,nskip=0,thin=1):
    '''flattens a chain (i.e collects results from all walkers), 
    with options to skip the first nskip parameters, and thin the chain
    by only retrieving a point every thin steps - thinning can be useful when
    the steps of the chain are highly correlated'''
    return chain[:,nskip::thin,:].reshape((-1,npars))
    
def readchain(file,nskip=0,thin=1):
    data = np.loadtxt(file)
    nwalkers=int(data[:,0].max()+1)
    nprod = int(data.shape[0]/nwalkers)
    npars = data.shape[1]-1 # first is walker ID, last is ln_prob
    chain = np.reshape(data[:,1:],(nwalkers,nprod,npars))
    return chain

def plotchains(chain,npar,alpha=0.2):
    nwalkers, nsteps, npars = chain.shape
    fig = plt.figure()
    for i in range(nwalkers):
        plt.plot(chain[i,:,npar],alpha=alpha,color='k')
    return fig

def rebin(xbins,x,y,e=None,weighted=True,errors_from_rms=False):
    digitized = np.digitize(x,xbins)
    xbin = []
    ybin = []
    ebin = []
    for i in range(0,len(xbins)):
            bin_y_vals = y[digitized == i]
            bin_x_vals = x[digitized == i]
            if e is not None:
                bin_e_vals = e[digitized == i]
            if weighted:
                if e is None:
                    raise Exception('Cannot compute weighted mean without errors')
                weights = 1.0/bin_e_vals**2
                xbin.append( np.sum(weights*bin_x_vals) / np.sum(weights) )
                ybin.append( np.sum(weights*bin_y_vals) / np.sum(weights) )
                if errors_from_rms:
                    ebin.append(np.std(bin_y_vals))
                else:
                    ebin.append( np.sqrt(1.0/np.sum(weights) ) )
            else:
                xbin.append(bin_x_vals.mean())
                ybin.append(bin_y_vals.mean())
                if errors_from_rms:
                    ebin.append(np.std(bin_y_vals))
                else:
                    ebin.append(np.sqrt(np.sum(bin_e_vals**2)) / len(bin_e_vals))
    xbin = np.array(xbin)
    ybin = np.array(ybin)
    ebin = np.array(ebin)
    return (xbin,ybin,ebin)