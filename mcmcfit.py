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
    
    lnp = 0.0
    
    prior = Prior('uniform',-15,10)
    lnp += prior.ln_prob(lna)
    prior = Prior('uniform',-7.97,-6.57) #flickering timescale 30s to 2 mins
    lnp += prior.ln_prob(lntau)
    
    return lnp + ln_prior_base(params[2:])

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
    prior = Prior('uniform',0.0,0.5)
    lnp += prior.ln_prob(pars[3])

    #Mass ratio
    prior = Prior('uniform',0.001,3.5)
    lnp += prior.ln_prob(pars[4])

    #Wd eclipse width, dphi
    tol = 1.0e-6
    try:
        maxphi = roche.findphi(pars[4],90.0) #dphi when i is slightly less than 90
        prior = Prior('uniform',0.001,maxphi-tol)
        lnp += prior.ln_prob(pars[5])
    except:
        # we get here when roche.findphi raises error - usually invalid q
        lnp += -np.inf

    #Disc radius (XL1) 
    try:
        xl1 = roche.xl1(pars[4]) # xl1/a
        prior = Prior('uniform',0.25,0.46/xl1) # maximum size disc can be without precessing
        lnp += prior.ln_prob(pars[6])
    except:
        # we get here when roche.findphi raises error - usually invalid q
        lnp += -np.inf
    
    #Limb darkening
    prior = Prior('gauss',0.35,0.005)
    lnp += prior.ln_prob(pars[7])

    #Wd radius (XL1)
    prior = Prior('uniform',0.0001,0.1)
    lnp += prior.ln_prob(pars[8])

    #BS scale (XL1)
    rwd = pars[8]
    prior = Prior('log_uniform',rwd/3.,rwd*3.)
    lnp += prior.ln_prob(pars[9])

    #BS az
    slop=40.0

    try:
        # find position of bright spot where it hits disc
        # will fail if q invalid
        xl1 = roche.xl1(pars[4]) # xl1/a
        rd_a = pars[6]*xl1

        # Does stream miss disc? (disc/a < 0.2 or > 0.65 )
        # if so, Tom's code will fail
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
        prior = Prior('uniform',0.9,3.0)
        lnp += prior.ln_prob(pars[15])

        #BS tilt angle
        prior = Prior('uniform',0.0,165.0)
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

def calcWdEclipseMask(dphi, phiOff,phi):
    # calculate mask which selects in eclipse points
    phiStart = 1-dphi/2+phiOff
    phiEnd   = dphi/2 + phiOff
    fracPhi  = phi % 1
    return (fracPhi < phiEnd) | (fracPhi > phiStart)
    
def kernelCalc(x1,x2,pars):
    '''This is a function that evaluates the kernel function 
    given arguments (x1, x2, p) where x1 and x2 are numpy array 
    defining the coordinates of the samples and p is the numpy 
    array giving the current settings of the parameters.
    
    In this case it is a simple expSquaredKernel, except that 
    we implement two changepoints at times corresponding to wd
    eclipse (http://www.robots.ox.ac.uk/~parg/pubs/changepoint.pdf)
    
    We assume the points across changepoints are uncorrelated, 
    and that the amplitude of the GP inside eclipse is very small'''
    amp, tau, dphi, phi0 = pars
    kernel='Matern32'

    # use numpy broadcasting to create 2d array of time differences between x1 and x2
    rij = (x1[:,np.newaxis]-x2[np.newaxis,:])**2 / tau

    # calculate masks which select only those points in eclipse
    mask1 = calcWdEclipseMask(dphi,phi0,x1)
    mask2 = calcWdEclipseMask(dphi,phi0,x2)
    
    # use the same broadcasting trick to have a 2d mask which  
    # true if both points are in eclipse, or if either point is
    bothEclipsed = np.logical_and(mask1[:,np.newaxis] , mask2[np.newaxis,:])
    oneEclipsed  = np.logical_xor(mask1[:,np.newaxis] , mask2[np.newaxis,:])
    
    # calculate kernel
    if kernel == 'ExpSquared':
        vij = amp*np.exp(-0.5*rij)
    elif kernel == 'Matern32':
        vij = amp*(1.0+np.sqrt(3*rij))*np.exp(-np.sqrt(3*rij))
            
    # normal amplitude if both out of eclipse
    # reduced amplitude for both in eclipse
    # zero for one in eclipse, one not
    vij[oneEclipsed] = 0.0
    vij[bothEclipsed] = vij[bothEclipsed]/50.
    return vij 

def createGP(params,phi):
    a, tau = np.exp(params[:2])
    dphi, phiOff = params[7],params[15]
    
    # custom kernel with changepoints at WD eclipse
    kernel = kernels.PythonKernel(kernelCalc,\
        pars=(a,tau,dphi,phiOff))

    # create GPs using this kernel
    gp  = george.GP(kernel)
    
    return gp
        
def lnlike_gp(params, phi, width, y, e, cv): 
    gp = createGP(params,phi)
    gp.compute(phi,e)
    
    resids = y - model(params[2:],phi,width,cv)
    
    # check for bugs in model
    if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
        print params
        print 'Warning: model gave nan or inf answers'
        #raise Exception('model gave nan or inf answers')
        return -np.inf
 
    # now calculate ln_likelihood
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
    
    # create GP with changepoints at WD eclipse
    gp = createGP(bestFit,x)
    gp.compute(x,e)
    
    #condition GP on residuals, and draw conditional samples
    samples = gp.sample_conditional(res, x, size=300)
    # compute mean and standard deviation of samples from GPs
    mu      = np.mean(samples,axis=0)
    std     = np.std(samples,axis=0)

    # don't forget to predict at fine samples
    fmu, _ = gp.predict(res, xf)

    #set up plot subplots
    gs = gridspec.GridSpec(3,1,height_ratios=[2,1,1])
    gs.update(hspace=0.0)
    ax_dat = plt.subplot(gs[0,0])
    ax_dat_mod = plt.subplot(gs[1,0],sharex=ax_dat)
    ax_res = plt.subplot(gs[2,0],sharex=ax_dat) 
    
    #data plot
    ax_dat.plot(xf,yf+fmu,'r-')
    ax_dat.errorbar(x,y,yerr=e,fmt='.',color='k',capsize=0)

   
    #data - model plot
    ax_dat_mod.plot(xf,yf,'r-')
    ax_dat_mod.errorbar(x,y-mu,yerr=e,fmt='.',color='k',capsize=0)
    
    #residual plot
    ax_res.errorbar(x,res,yerr=e,fmt='.',color='k',capsize=0)
    ax_res.fill_between(x,mu+2.0*std,mu-2.0*std,color='r',alpha=0.4)

    #fix the x-axes
    plt.setp(ax_dat.get_xticklabels(),visible=False)
    ax_res.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4,prune='both'))
    ax_dat.set_ylabel('Flux (mJy)')
    ax_dat_mod.set_ylabel('Flux (mJy)')
    ax_res.set_xlabel('Orbital Period')
    ax_res.set_ylabel('Residuals')
    plt.xlim(-0.1,0.15)
    plt.savefig('bestFit.pdf')
    plt.show()
    
def parseStartPars(file):
    f = open(file)
    newDict = {}
    for line in f:
        k,v = line.strip().split('=')
        newDict[k.strip()] = float(v.strip())
    return newDict
    
if __name__ == "__main__":


    #Input lightcurve data from txt file
    import argparse
    parser = argparse.ArgumentParser(description='Fit CV lightcurves with lfit')
    parser.add_argument('file',action='store',help='input file (x,y,e)')
    parser.add_argument('parfile',action='store',help='starting parameters')
    parser.add_argument('-f','--fit',action='store_true',help='actually fit, otherwise just plot')
    parser.add_argument('-nw','--nwalkers',action='store',help='number of walkers', type=int, default=100)
    parser.add_argument('-nt','--nthreads',action='store',help='number of threads', type=int, default=6)
    parser.add_argument('-nb','--nburn',action='store',help='number of burn steps', type=int, default=5000)
    parser.add_argument('-np','--nprod',action='store',help='number of prod steps', type=int, default=5000)
    args = parser.parse_args()
    file = args.file
    toFit = args.fit

    x,y,e = np.loadtxt(file,skiprows=16).T
    width = np.mean(np.diff(x))*np.ones_like(x)/2.
    
    parDict = parseStartPars(args.parfile)
    q = parDict['q']
    dphi = parDict['dphi']
    rwd = parDict['rwd']
    ulimb = parDict['ulimb']
    rdisc = parDict['rdisc']
    rexp = parDict['rexp']
    az = parDict['az']
    frac = parDict['frac']
    scale = parDict['scale']
    fwd = parDict['fwd']
    fdisc = parDict['fdisc']
    fbs = parDict['fbs']
    fd = parDict['fd']
    off = parDict['off']
    amp_gp = parDict['amp_gp']
    tau_gp = parDict['tau_gp']
    try:
        exp1 = parDict['exp1']
        exp2 = parDict['exp2']
        tilt = parDict['tilt']
        yaw = parDict['yaw']    
        guessP = np.array([amp_gp,tau_gp,fwd,fdisc,fbs,fd,q,dphi,rdisc,ulimb,rwd,scale,az,frac,rexp,off, \
                          exp1,exp2,tilt,yaw])
    except:
        guessP = np.array([amp_gp,tau_gp,fwd,fdisc,fbs,fd,q,dphi,rdisc,ulimb,rwd,scale,az,frac,rexp,off])
        
    # is our starting position legal
    if np.isinf( ln_prior_gp(guessP) ):
        print parDict
        print 'Error: starting position violates priors'
        sys.exit(-1)
        
    # initialize a cv with these params
    myCV = lfit.CV(guessP[2:])
    
    lnprior = ln_prior_gp(guessP)
    lnlikelihood = lnlike_gp(guessP, x, width, y, e, myCV)
    lnprob = lnprob_gp(guessP, x, width, y, e, myCV)
    
    print "lnprior =      " ,lnprior
    print "lnlikelihood = " ,lnlikelihood
    print "lnprob =       " ,lnprob

    if toFit:
        npars = len(guessP)
       
        mcmcPars = (args.nwalkers,args.nburn,args.nburn,args.nprod,args.nthreads)
        sampler = fit_gp(guessP, x, width, y, e, myCV, mcmcPars)

        chain = flatchain(sampler.chain,npars,thin=5)
        nameList = ['lna','lntau','fwd','fdisc','fbs','fd','q','dphi','rdisc','ulimb','rwd','scale', \
                    'az','frac','rexp','off','exp1','exp2','tilt','yaw']
        bestPars = []
        for i in range(npars):
            par = chain[:,i]
            lolim,best,uplim = np.percentile(par,[16,50,84])
            print "%s = %f +%f -%f" % (nameList[i],best,uplim-best,best-lolim)
            bestPars.append(best)
        fig = thumbPlot(chain,nameList,truths=guessP)
        fig.savefig('cornerPlot.pdf')
        plt.close()
    else:
        bestPars = guessP

    plot_result(bestPars, x, width, y, e, myCV)
    
