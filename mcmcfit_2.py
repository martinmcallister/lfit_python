import numpy as np
import matplotlib.pyplot as plt
from trm import roche
import sys
import lfit
import emcee
import george
from george import kernels
from mcmc_utils import *


def model(pars,phi,width,cv):
    '''
    Model params:
    fwd - white dwarf flux
    fdisc - disc flux
    fbs - bright spot flux
    fd - donor flux
    q - mass ratio
    dphi - width of wd eclipse
    rdisc - disc radius (rdisc/xl1)
    ulimb - limb darkening of wd
    rwd - primary radius (r1/xl1)
    az,frac,scale - compulsory bright spot params
    rexp - disc exponent
    off - phase offset
    [exp1,exp2,tilt,yaw] - [optional] bright spot params
    '''
    # CV takes a list of params which *must* include
    # fwd,fdisc,fbs,fd,q,dphi,rdisc (xl1), ulimb, rwd (xl1), scale, az, frac, rexp and phi0
    # if that's it. It makes a simple bright spot

    # if you provide a longer list which has at the end
    # exp1,exp2,tilt,yaw
    # it uses a complex bright spot

    # once you've made a CV, you can update the params when you call calcFlux
    # and the same is true
    return cv.calcFlux(pars,phi,width)

def model_gp(params, x):
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

def lnprior_gp(params):
    lna, lntau = params[:2]
    if not -5 < lna < 5:
        return -np.inf
    if not -5 < lntau < 5:
        return -np.inf
    return lnprior_base(params[2:])

def ln_prior(pars):

    lnp = 0.0

     #Wd flux
    prior = Prior('uniform',0.001,0.5)
    lnp += prior.ln_prob(pars[0])

    #Disc flux
    prior = Prior('uniform',0.001,0.5)
    lnp += prior.ln_prob(pars[1])

    #BS flux
    prior = Prior('uniform',0.001,0.5)
    lnp += prior.ln_prob(pars[2])

    #Donor flux
    prior = Prior('uniform',0.0,0.01)
    lnp += prior.ln_prob(pars[3])

    #Mass ratio
    prior = Prior('uniform',0.001,3.5)
    lnp += prior.ln_prob(pars[4])

    #Wd eclipse width, dphi
    tol = 1.0e-6
    maxphi = roche.findphi(pars[4],90.0) #dphi when i is slightly less than 90
    prior = Prior('uniform',0.001,maxphi-tol)
    lnp += prior.ln_prob(pars[5])

    #Disc radius (XL1) 
    prior = Prior('uniform',0.3,0.9)
    lnp += prior.ln_prob(pars[6])
    
    #Limb darkening
    prior = Prior('gauss',0.35,0.005)
    lnp += prior.ln_prob(pars[7])

    #Wd radius (XL1)
    prior = Prior('uniform',0.0001,0.1)
    lnp += prior.ln_prob(pars[8])

    #BS scale (XL1)
    rwd = pars[8]
    prior = Prior('uniform',rwd/3.,0.5)
    lnp += prior.ln_prob(pars[9])

    #BS az
    slop=40.0
    # find position of bright spot where it hits disc
    xl1 = roche.xl1(pars[4]) # xl1/a
    rd_a = pars[6]*xl1
    # Does stream miss disc? (disc/a < 0.2 or > 0.65 )
    # if so, Tom's code will fail
    try:
       x,y,vx,vy = roche.bspot(pars[4],rd_a)
            
       # find tangent to disc at this point
       alpha = np.degrees(np.arctan2(y,x))
            
       # alpha is between -90 and 90. if negative spot lags disc ie alpha > 90
       if alpha < 0: alpha = 90-alpha
       tangent = alpha + 90 # disc tangent
    
       prior = Prior('uniform',max(0,tangent-slop),min(180,tangent+slop))
       lnp += prior.ln_prob(pars[10])
    except:
       lnp += -np.inf
       
    #BS isotropic fraction
    prior = Prior('uniform',0.001,0.9)
    lnp += prior.ln_prob(pars[11])
    
    #Disc exponent
    prior = Prior('uniform',0.0001,2.5)
    lnp += prior.ln_prob(pars[12])

    #Phase offset
    prior = Prior('uniform',-0.1,0.1)
    lnp += prior.ln_prob(pars[13])

    if len(pars) > 14:
        #BS exp1
        prior = Prior('uniform',0.01,4.0)
        lnp += prior.ln_prob(pars[14])

        #BS exp2
        prior = Prior('uniform',0.01,3.0)
        lnp += prior.ln_prob(pars[15])

        #BS tilt angle
        prior = Prior('uniform',0.0,170.0)
        lnp += prior.ln_prob(pars[16])

        #BS yaw angle
        prior = Prior('uniform',-90,90.0)
        lnp += prior.ln_prob(pars[17])
    return lnp

def chisq(y,yfit,e):
    resids = ( y - yfit ) / e
    return np.sum(resids*resids)

def reducedChisq(y,yfit,e,pars):
    return chisq(y,yfit, e) / (len(y) - len(pars) - 1)

def lnlike_gp(params, x, y, e):
    a, tau = np.exp(params[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(x, e)
    return gp.lnlikelihood(y - model_gp(params[2:], x))

def ln_likelihood(pars,phi,width,y,e,cv):
    yfit = model(pars,phi,width,cv)
    return -0.5*(np.sum( np.log( 2.0*np.pi*e**2 ) ) + chisq(y,yfit,e))

def ln_prob(pars,phi,width,y,e,cv):
    lnp = ln_prior(pars)
    if np.isfinite(lnp):
        return lnp + ln_likelihood(pars,phi,width,y,e,cv)
    else:
        return lnp

def lnprob_gp(params, x, y, e):
    lp = lnprior_gp(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(params, x, y, e)

def fit_gp(initial, data, nwalkers):
    ndim_gp = len(initial)
    p0_gp = [np.array(initial) + 1e-8 * np.random.randn(ndim_gp)
          for i in xrange(nwalkers)]
    sampler_gp = emcee.EnsembleSampler(nwalkers, ndim_gp, lnprob_gp, args=data, threads=6)

    print("Running burn-in")
    p0_gp, lnp, _ = sampler_gp.run_mcmc(p0_gp, 500)
    sampler_gp.reset()

    print("Running second burn-in")
    params = p0_gp[np.argmax(lnp)]
    p0_gp = [params + 1e-8 * np.random.randn(ndim_gp) for i in xrange(nwalkers)]
    p0_gp, _, _ = sampler_gp.run_mcmc(p0_gp, 500)
    sampler_gp.reset()

    print("Running production")
    p0_gp, _, _ = sampler_gp.run_mcmc(p0_gp, 1000)

    return sampler_gp

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
    
    q = 0.087049
    dphi = 0.053844
    rwd = 0.027545
    ulimb = 0.35
    rdisc = 0.3788224
    rexp = 0.349158
    az = 164.16167
    frac = 0.139078
    scale = 0.0231764
    exp1 = 2.0
    exp2 = 1.0
    tilt = 60.0
    yaw = 1.0
    fwd = 0.128650
    fdisc = 0.048163
    fbs = 0.0745461
    fd = 0.001
    off = -0.000078024

    guessP = np.array([fwd,fdisc,fbs,fd,q,dphi,rdisc,ulimb,rwd,scale,az,frac,rexp,off, \
                      exp1,exp2,tilt,yaw])

    # is our starting position legal
    if np.isinf( ln_prior(guessP) ):
        print 'Error: starting position violates priors'
        sys.exit(-1)
        
    # initialize a cv with these params
    myCV = lfit.CV(guessP)

    if toFit:
    
        npars = len(guessP)
        nwalkers = 50
        nthreads = 1
        p0 = emcee.utils.sample_ball(guessP,0.05*guessP,size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[x,width,y,e,myCV],threads=nthreads)

        #Burn-in
        nburn = 50
        pos, prob, state = run_burnin(sampler,p0,nburn)

    
        #Production
        sampler.reset()
        nprod = 100
        sampler = run_mcmc_save(sampler,pos,nprod,state,"chain.txt")  
        chain = flatchain(sampler.chain,npars,thin=4)
        
        nameList = ['fwd','fdisc','fbs','fd','q','dphi','rdisc','ulimb','rwd','scale', \
                    'az','frac','rexp','off','exp1','exp2','tilt','yaw']
        bestPars = []
        for i in range(npars):
            par = chain[:,i]
            lolim,best,uplim = np.percentile(par,[16,50,84])
            print "%s = %f +%f -%f" % (nameList[i],best,uplim-best,best-lolim)
            bestPars.append(best)
        fig = thumbPlot(chain,nameList)
        fig.savefig('cornerPlot.pdf')
        plt.close()
    else:
        bestPars = guessP
        
    #Fit assuming GP
    print("Fitting assuming GP")
    data = (x, y, e)
    truth = [0.0, 0.0, 0.0]
    truth_gp = [0.0, 0.0] + truth
    sampler_gp = fit_gp(truth_gp, data, 16)
    
    #Plot gp model & data
    print("Making gp plot")
    samples = sampler_gp.flatchain
    xf = np.linspace(x.min(),x.max(),1000)
    wf = 0.5*np.mean(np.diff(xf))*np.ones_like(xf)
    plt.figure()
    plt.subplot(3,1,(1,2))
    plt.errorbar(x,y,yerr=e,fmt='.',color='k',capsize=0)
    for s in samples[np.random.randint(len(samples), size=24)]:
        gp = george.GP(np.exp(s[0])*kernels.Matern32Kernel(np.exp(s[1])))
        gp.compute(x,e)
        yf_gp = gp.sample_conditional(y - model_gp(s[2:], x), xf) + model_gp(s[2:], xf)
        plt.plot(xf,yf_gp,'r-')
    #plt.xlabel('Orbital Phase')
    plt.ylabel('Flux')
    plt.xlim(-0.1,0.15)
    
    #Plot cv model & data
    plt.subplot(3,1,(1,2))
    yf = model(bestPars,xf,wf,myCV)
    plt.plot(xf,yf,'b-')
    plt.errorbar(x,y,yerr=e,fmt='.',color='k',capsize=0)
    #plt.xlabel('Orbital Phase')
    plt.ylabel('Flux')
    plt.xlim(-0.1,0.15)

    plt.subplot(3,1,3)
    plt.plot(xf,yf_gp-yf,'k-')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Flux')
    plt.xlim(-0.1,0.15)
    plt.show()
    
