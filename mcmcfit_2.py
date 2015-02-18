import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
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


def ln_prior_gp(params):
    lna, lntau = params[:2]
    if not -5 < lna < 5:
        return -np.inf
    if not -10 < lntau < 10:
        return -np.inf
    return ln_prior_base(params[2:])

def ln_prior_base(pars):

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
    
       prior = Prior('uniform',max(0,tangent-slop),min(178,tangent+slop))
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

def lnlike_gp(params, phi, width, y, e, cv):
    a, tau = np.exp(params[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(phi, e)
    resids = y - model(params[2:],phi,width,cv)
    if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
        print params[2:]
        raise Exception('model gave nan or inf answers')
    return gp.lnlikelihood(resids)

def lnprob_gp(params, phi, width, y, e, cv):
    lp = ln_prior_gp(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(params, phi, width, y, e, cv)

def fit_gp(initialGuess, phi, width, y, e, cv, mcmcPars=(100,300,300,500,6)):
    nwalkers, nBurn1, nBurn2, nProd, nThreads = mcmcPars
    ndim_gp = len(initialGuess)
    pos = emcee.utils.sample_ball(initialGuess,0.01*initialGuess,size=nwalkers)
    sampler_gp = emcee.EnsembleSampler(nwalkers, ndim_gp, lnprob_gp, args=(phi,width,y,e,cv), threads=nThreads)

    print("Running burn-in")
    pos, prob, state = run_burnin(sampler_gp,pos,nBurn1)
    sampler_gp.reset()

    print("Running second burn-in")
    # choose the highest probability point in first Burn-In as starting point
    pos = pos[np.argmax(prob)]
    pos = emcee.utils.sample_ball(pos,0.01*pos,size=nwalkers)
    pos, prob, state = run_burnin(sampler_gp,pos,nBurn2)
    sampler_gp.reset()

    print("Running production")
    sampler_gp = run_mcmc_save(sampler_gp,pos,nProd,state,"chain.txt")

    return sampler_gp

def plot_result(bestFit, x, width, y, e, cv):

    #calc residuals
    fit = model(bestFit[2:],x,width,cv)
    res = y-fit

    #fine scale fit for plotting
    xf = np.linspace(x.min(),x.max(),1000)
    wf = 0.5*np.mean(np.diff(xf))*np.ones_like(xf)
    yf = model(bestFit[2:],xf,wf,cv)
    
    # GP
    a,tau = np.exp(bestFit[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(x,e)

    # condition GP on residuals, and draw conditional samples
    samples = gp.sample_conditional(res, x, size=300)
    mu      = np.mean(samples,axis=0)
    std     = np.std(samples,axis=0)

    # set up plot subplots
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    gs.update(hspace=0.0)
    ax_main = plt.subplot(gs[0,0])
    ax_res = plt.subplot(gs[1,0],sharex=ax_main)

    #main plot
    ax_main.plot(xf,yf,'k-')
    ax_main.errorbar(x,y-mu,yerr=e,fmt='.',color='k',capsize=0)

    #residual plot
    ax_res.errorbar(x,res,yerr=e,fmt='.',color='k',capsize=0)
    ax_res.fill_between(x,mu+2.0*std,mu-2.0*std,color='r',alpha=0.4)

    #fix the x-axes
    plt.setp(ax_main.get_xticklabels(),visible=False)
    ax_res.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4,prune='both'))
    ax_res.set_xlabel('MJD (days)')
    ax_res.set_ylabel('Residuals')
    ax_main.set_ylabel('Flux (mJy)')
    plt.savefig('bestFit.pdf')
    
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

    #also need hyperparameters for GP
    amp_gp = np.log( 0.05*y.mean() ) #5% of mean flux
    tau_gp = np.log( 60./86400. ) #one minute
    
    guessP = np.array([amp_gp,tau_gp,fwd,fdisc,fbs,fd,q,dphi,rdisc,ulimb,rwd,scale,az,frac,rexp,off, \
                      exp1,exp2,tilt,yaw])

    # is our starting position legal
    if np.isinf( ln_prior_gp(guessP) ):
        print 'Error: starting position violates priors'
        sys.exit(-1)
        
    # initialize a cv with these params
    myCV = lfit.CV(guessP[2:])

    if toFit:
        npars = len(guessP)
        nwalkers = 50
        nthreads = 4
        nburn = 50
        nprod = 100
        mcmcPars = (nwalkers,nburn,nburn,nprod,nthreads)
        sampler = fit_gp(guessP, x, width, y, e, myCV, mcmcPars)

        chain = flatchain(sampler.chain,npars,thin=20)
        nameList = ['lna','lntau','fwd','fdisc','fbs','fd','q','dphi','rdisc','ulimb','rwd','scale', \
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

    plot_result(bestPars, x, width, y, e, myCV)
    
