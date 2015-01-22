from scipy import optimize as opt
import numpy as np
import emcee
import mcmc_utils
from astropy import coordinates, units
#from astropy_utils import *
import matplotlib.pyplot as plt

def model(pars,x):
    '''model parameters are
    p0 - mean level
    p1 - amplitude of sine wave
    p2 - phase offset of sine wave
    p3 - additive noise (see ln_like)
    ''' 
    phase = pars[2] + x
    # irradiation curve should have minimum at PHI=0.5,
    # so described by sin(2*pi*phi-0.5)
    return pars[0] +  pars[1]*np.sin( 2.0*np.pi*(phase-0.25) ) 
    
def chisq(pars,x,y,yerr):
    resids = ( y - model(pars,x) ) / yerr
    return np.sum(resids*resids)
    
def reducedChisq(pars,x,y,yerr):
    return chisq(pars,x,y,yerr) / (len(x) - len(pars) - 1)

def ln_prior(pars):
    lnp = 0.0
    # noise - uniform in log with min 0.0001 and max 0.1
    prior = mcmc_utils.Prior('log_uniform',0.0001,0.1)
    lnp += prior.ln_prob(pars[3])
    
    # phase - uniform between 0 and 1
    prior = mcmc_utils.Prior('uniform',-0.2,0.2)
    lnp += prior.ln_prob(pars[2])

    # amplitude of sine wave, enforce positive (uniform between 0 and 10)
    prior = mcmc_utils.Prior('uniform',0.0,10)
    lnp += prior.ln_prob(pars[1])

    return lnp

def ln_likelihood(pars,x,y,yerr):
    errs        = np.sqrt(yerr**2 + pars[3]**2)
    return -0.5*(np.sum( np.log( 2.0*np.pi*errs**2 ) ) + chisq(pars,x,y,errs))
    
def ln_prob(pars,x,y,yerr): 
    #return ln_prior(pars) + ln_likelihood(pars,x,y,yerr)
    lnp = ln_prior(pars)
    if np.isfinite(lnp):
        return lnp + ln_likelihood(pars,x,y,yerr)
    else:
        return lnp    
      

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fit or plot model of LC')
    parser.add_argument('file',help='txt file with LC (phase,flux,err)')
    parser.add_argument('--nwalkers',action='store',type=int,default=40)
    parser.add_argument('--nburn',action='store',type=int,default=200)
    parser.add_argument('--nprod',action='store',type=int,default=500)
    parser.add_argument('--nthreads',action='store',type=int,default=4)
    args = parser.parse_args()
    
    x,y,e = np.loadtxt(args.file,skiprows=16).T
    xmean = x.mean()
    hjd0 = np.random.normal(54178.1762,0.0001,size=1e5)
    per  = np.random.normal(0.07943002,0.00000003,size=1e5)
    pmean = (xmean-hjd0)/per
    print 'Phase uncertainty = %f' % pmean.std()

    # convert to phase
    hjd0 = 54178.1762
    per  = 0.07943002
    x = (x-hjd0)/per 
    x -= np.floor(x)
    
    # mean level, amp of sin, phase, noise
    guessP = np.array([ y.mean(), y.mean()*0.02, 0.05, np.fabs(0.5*e.mean())])
    npars = len(guessP)
    nwalkers = args.nwalkers
    p0 = emcee.utils.sample_ball(guessP,0.02*guessP,size=nwalkers)
    
    sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[x,y,e],threads=args.nthreads)
    
    #burn-in
    nburn = args.nburn
    pos, prob, state = mcmc_utils.run_burnin(sampler, p0,nburn)
    
    #production
    sampler.reset()
    nprod = args.nprod
    sampler = mcmc_utils.run_mcmc_save(sampler,pos,nprod,state,"chain.txt")

    chain = mcmc_utils.flatchain(sampler.chain,npars,thin=3)
    nameList = ['Mean','Amp','Phi','Noise']
    bestPars = []
    for i in range(npars):
        par = chain[:,i]
        lolim,best,uplim = np.percentile(par,[16,50,84])
        print "%s = %f +%f -%f" % (nameList[i],best,uplim-best,best-lolim)
        bestPars.append(best)

    fig = mcmc_utils.thumbPlot(chain,nameList)
    fig.savefig('cornerPlot.pdf')
    plt.clf()
    
    xf = np.linspace(0,2,1000)
    yf = model(bestPars,xf)
    plt.plot(xf,yf,'k-')
    plt.errorbar(x,y,yerr=e,fmt='.',color='k')
    plt.errorbar(x+1,y,yerr=e,fmt='.',color='k')
    plt.xlabel('Orbital Phase (RV)')
    plt.ylabel('Magnitudes')
    plt.gca().invert_yaxis()
    plt.savefig('bestFit.pdf')    
    plt.clf()
