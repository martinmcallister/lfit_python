import numpy as np
import matplotlib.pyplot as plt
from trm import roche
import sys
import lfit
import emcee
import george
from george import kernels
from mcmc_utils import *

def model(params, x):
    amp, loc, sig2 = params
    return amp * np.exp(-0.5 * (x - loc) ** 2 / sig2)

def lnprior_base(params):
    amp, loc, sig2 = params
    if not -10 < amp < 10:
        return -np.inf
    if not -5 < loc < 5:
        return -np.inf
    if not 0 < sig2 < 3.0:
        return -np.inf
    return 0.0

def lnlike_gp(params, x, y, e):
    a, tau = np.exp(params[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(x, e)
    return gp.lnlikelihood(y - model(params[2:], x))

def lnprior_gp(params):
    lna, lntau = params[:2]
    if not -5 < lna < 5:
        return -np.inf
    if not -5 < lntau < 5:
        return -np.inf
    return lnprior_base(params[2:])


def lnprob_gp(params, x, y, e):
    lp = lnprior_gp(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(params, x, y, e)


def fit_gp(initial, data, nwalkers):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running second burn-in")
    params = p0[np.argmax(lnp)]
    p0 = [params + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler


if __name__ == "__main__":


    #Input lightcurve data from txt file
    import argparse
    parser = argparse.ArgumentParser(description='Fit CV lightcurves with lfit')
    parser.add_argument('file',action='store',help='input file (x,y,e)')
    parser.add_argument('-f','--fit',action='store_true',help='actually fit, otherwise just plot')
    args = parser.parse_args()
    file = args.file
    toFit = args.fit

    x,y,e = np.loadtxt(file,skiprows=16).T
    width = np.mean(np.diff(x))*np.ones_like(x)/2.
    
    # Fit assuming GP
    print("Fitting assuming GP")
    data = (x, y, e)
    truth = [0.0, 0.0, 0.0]
    truth_gp = [0.0, 0.0] + truth
    sampler = fit_gp(truth_gp, data, 16)
    
    #Plot model & data
    print("Making plots")
    samples = sampler.flatchain
    xf = np.linspace(x.min(),x.max(),1000)
    wf = 0.5*np.mean(np.diff(xf))*np.ones_like(xf)
    plt.figure()
    plt.errorbar(x,y,yerr=e,fmt='.',color='k',capsize=0)
    for s in samples[np.random.randint(len(samples), size=24)]:
        gp = george.GP(np.exp(s[0])*kernels.Matern32Kernel(np.exp(s[1])))
        gp.compute(x,e)
        yf = gp.sample_conditional(y - model(s[2:], x), xf) + model(s[2:], xf)
        plt.plot(xf,yf,'r-')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Flux')
    plt.savefig('bestFit.pdf')
    plt.xlim(-0.1,0.15)
    plt.show()
'''
    # Make the corner plot.
    fig = triangle.corner(samples[:, 2:], truths=truth)
    fig.savefig("gp-corner.pdf", dpi=150)
'''
