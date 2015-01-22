import lfit
import numpy as np
import matplotlib.pyplot as plt
from mcmc_utils import *
import emcee
from trm import roche

def model(pars,x):
    '''pars are 
    mass ratio (shouldn't matter unless donor is close to RL filling)
    size of primary r1_a
    limb darkening of wd
    size of donor r2_a
    inclination 
    wd flux
    donor flux
    phase offset'''
    q, r1_a, ulimb, r2_a, incl, wdNorm, donorNorm, xoff= pars
    wd = lfit.PyWhiteDwarf(r1_a,ulimb)
    phi = np.array(x-xoff)
    width = np.mean(np.diff(phi))*np.ones_like(phi)/2.
    wdCurve = wd.calcFlux(q,incl,phi,width)
    return donorNorm + wdNorm*wdCurve
 
def ln_prior(pars):
    lnp = 0.0
    # mass ratio - be loose (B12 says about 0.15)
    prior = Prior('uniform',0.02,0.3)
    lnp += prior.ln_prob(pars[0])
    # r1_a (B12)
    prior = Prior('gaussPos',0.0213,0.0015)
    lnp += prior.ln_prob(pars[1])
    # limb darkening
    prior = Prior('gaussPos',0.32,0.03)
    lnp += prior.ln_prob(pars[2])
    # size of donor (B12)
    prior = Prior('gaussPos',0.113,0.02)
    #is donor bigger than roche lobe?
    if (pars[0] <= 0 or pars[3] > 1.0-roche.xl1(pars[0])):
        lnp += -np.inf
    else:
        lnp += prior.ln_prob(pars[3])
    # inclination (B12)
    prior = Prior('gaussPos',85.9,1.0)
    if pars[4] >= 90.0:
        lnp += -np.inf
    else:
        lnp += prior.ln_prob(pars[4])
    # wd flux
    prior = Prior('uniform',0.01,0.05)
    lnp += prior.ln_prob(pars[5])
    # donor flux
    prior = Prior('uniform',0.00,0.01)
    lnp += prior.ln_prob(pars[6])
    #phase offset
    prior = Prior('uniform',-0.01,0.01)
    lnp += prior.ln_prob(pars[7])
    return lnp
    
def chisq(pars,x,y,yerr):
    phi = np.linspace(x[0],x[len(x)-1],100)
    resids = ( y - model(pars,x) ) / yerr
    return np.sum(resids*resids)
    
def reducedChisq(pars,x,y,yerr):
    return chisq(pars,x,y,yerr) / (len(x) - len(pars) - 1)

def ln_likelihood(pars,x,y,yerr):
    errs        = yerr
    return -0.5*(np.sum( np.log( 2.0*np.pi*errs**2 ) ) + chisq(pars,x,y,errs))

def ln_prob(pars,x,y,yerr): 
    #return ln_prior(pars) + ln_likelihood(pars,x,y,yerr)
    lnp = ln_prior(pars)
    if np.isfinite(lnp):
        return lnp + ln_likelihood(pars,x,y,yerr)
    else:
        return lnp

if __name__ == "__main__":

    import sys
    file = sys.argv[1]
    x,y,e = np.loadtxt(file,skiprows=16).T
    
    #remove integer phase
    x=x-np.floor(x)
    x[x>0.5] -= 1
    # get rid of data
    #mask = (x>-0.1) & (x<0.1)
    #x = x[mask]
    #y = y[mask]
    #e = e[mask]
            
    q = 0.15
    r1_a = 0.0213
    ulimb = 0.32
    r2_a = 0.113
    incl=85.9
    wdNorm = 0.02
    donorNorm = 0.00001
    phaseOff = 0.0000001
    guessP = np.array([q,r1_a,ulimb,r2_a,incl,wdNorm,donorNorm,phaseOff])
    npars = len(guessP)
    nwalkers = 100
    p0 = emcee.utils.sample_ball(guessP,0.01*guessP,size=nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[x,y,e],threads=2)
    
    #burnIn
    nburn = 1e2
    pos, prob, state = sampler.run_mcmc(p0,nburn)
    
    #production
    sampler.reset()
    nprod = 1e2
    sampler = run_mcmc_save(sampler,pos,nprod,state,"chain.txt")  
    chain = flatchain(sampler.chain,npars,thin=4)
    
    nameList = ['q','r1_a','U','r2_a','incl','wdFlux','donorFlux','phaseOff']
    bestPars = []
    for i in range(npars):
        par = chain[:,i]
        lolim,best,uplim = np.percentile(par,[16,50,84])
        print "%s = %f +%f -%f" % (nameList[i],best,uplim-best,best-lolim)
        bestPars.append(best)
    fig = thumbPlot(chain,nameList)
    fig.savefig('cornerPlot.pdf')
    plt.close()
    
    xf = np.linspace(x.min(),x.max(),1000)
    yf = model(bestPars,xf)
    plt.plot(xf,yf,'r-')
    plt.errorbar(x,y,yerr=e,fmt='.',color='k',capsize=0)
    plt.xlabel('Orbital Phase')
    plt.ylabel('Flux')
    plt.savefig('bestFit.pdf')
