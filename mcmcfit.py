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
    #prior = Prior('gauss',-6.437,0.001)
    lnp += prior.ln_prob(lna)
    prior = Prior('uniform',-7.97,-6.57) #flickering timescale 30s to 2 mins
    #prior = Prior('gauss',-7.925,0.001)
    lnp += prior.ln_prob(lntau)
    
    return lnp + ln_prior_base(params[2:])
    #return lnp + ln_prior_test(params[2:])

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
        prior = Prior('uniform',0.01,3.0)
        lnp += prior.ln_prob(pars[15])

        #BS tilt angle
        prior = Prior('uniform',0.0,165.0)
        lnp += prior.ln_prob(pars[16])

        #BS yaw angle
        prior = Prior('uniform',-90,90.0)
        lnp += prior.ln_prob(pars[17])
    return lnp
    
def ln_prior_test(pars):

    lnp = 0.0

    #Wd flux
    prior = Prior('gauss',0.139,0.001)
    lnp += prior.ln_prob(pars[0])

    #Disc flux
    prior = Prior('gauss',0.0586,0.0001)
    lnp += prior.ln_prob(pars[1])

    #BS flux
    prior = Prior('gauss',0.0478,0.0001)
    lnp += prior.ln_prob(pars[2])

    #Donor flux
    prior = Prior('gauss',0.00581,0.00001)
    lnp += prior.ln_prob(pars[3])

    #Mass ratio
    prior = Prior('gauss',0.08705,0.00001)
    lnp += prior.ln_prob(pars[4])

    #Wd eclipse width, dphi
    tol = 1.0e-6
    try:
        maxphi = roche.findphi(pars[4],90.0) #dphi when i is slightly less than 90
        prior = Prior('gauss',0.0538,0.0001)
        lnp += prior.ln_prob(pars[5])
    except:
        # we get here when roche.findphi raises error - usually invalid q
        lnp += -np.inf

    #Disc radius (XL1) 
    prior = Prior('gauss',0.405,0.001)
    lnp += prior.ln_prob(pars[6])
    
    #Limb darkening
    prior = Prior('gauss',0.35,0.005)
    lnp += prior.ln_prob(pars[7])

    #Wd radius (XL1)
    prior = Prior('gauss',0.01954,0.0001)
    lnp += prior.ln_prob(pars[8])

    #BS scale (XL1)
    prior = Prior('gauss',0.0330,0.0001)
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
    
        prior = Prior('gauss',165.1,0.1)
        lnp += prior.ln_prob(pars[10])
    except:
        lnp += -np.inf
        
    #BS isotropic fraction
    prior = Prior('gauss',0.414,0.001)
    lnp += prior.ln_prob(pars[11])
    
    #Disc exponent
    prior = Prior('gauss',0.292,0.001)
    lnp += prior.ln_prob(pars[12])

    #Phase offset
    prior = Prior('gauss',-0.00003,0.00001)
    lnp += prior.ln_prob(pars[13])

    if len(pars) > 14:
        #BS exp1
        prior = Prior('gauss',1.90,0.01)
        lnp += prior.ln_prob(pars[14])

        #BS exp2
        prior = Prior('gauss',0.90,0.01)
        lnp += prior.ln_prob(pars[15])

        #BS tilt angle
        prior = Prior('gauss',140.0,0.1)
        lnp += prior.ln_prob(pars[16])

        #BS yaw angle
        prior = Prior('gauss',-20.0,0.1)
        lnp += prior.ln_prob(pars[17])
    return lnp

def chisq(y,yfit,e):
    resids = ( y - yfit ) / e
    return np.sum(resids*resids)

def reducedChisq(y,yfit,e,pars):
    return chisq(y,yfit, e) / (len(y) - len(pars) - 1)

def lnlike_gp(params, phi, width, y, e, cv):
    a, tau = np.exp(params[:2])
    kernel = a * kernels.Matern32Kernel(tau)
    #kernel = a * kernels.ExpSquaredKernel(tau)
    gp = george.GP(kernel, solver=george.HODLRSolver)
    gp.compute(phi, e)
    resids = y - model(params[2:],phi,width,cv)
    if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
        print params
        print 'Warning: model gave nan or inf answers'
        #raise Exception('model gave nan or inf answers')
        return -np.inf
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
    #res_squared = res*res
    #mean_res_squared = np.mean(res_squared)
    #rms_res = mean_res_squared**0.5
    #print rms_res

    #fine scale fit for plotting
    xf = np.linspace(x.min(),x.max(),1000)
    wf = 0.5*np.mean(np.diff(xf))*np.ones_like(xf)
    yf = model(bestFit[2:],xf,wf,cv)
    
    #GP
    a,tau = np.exp(bestFit[:2])
    kernel = a * kernels.Matern32Kernel(tau)
    #kernel = a * kernels.ExpSquaredKernel(tau)
    gp = george.GP(kernel, solver=george.HODLRSolver)
    gp.compute(x,e)

    #condition GP on residuals, and draw conditional samples
    samples = gp.sample_conditional(res, x, size=300)
    mu      = np.mean(samples,axis=0)
    std     = np.std(samples,axis=0)
    fmu,_   = gp.predict(res,xf)

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
    ax_res.set_xlabel('MJD (days)')
    ax_res.set_ylabel('Residuals')
    plt.xlim(-0.1,0.15)
    plt.savefig('bestFit.pdf')
    plt.show()
    
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
    
    # PHL1445 average LC
    q = 0.087049
    dphi = 0.0538438
    rwd = 0.019544
    ulimb = 0.35
    rdisc = 0.40481
    rexp = 0.2924
    az = 165.101
    frac = 0.4144
    scale = 0.032956
    exp1 = 1.9
    exp2 = 0.9
    tilt = 140.0
    yaw = -20.0
    fwd = 0.139256
    fdisc = 0.05861
    fbs = 0.047829
    fd = 0.00581
    off = -0.00003
    
    '''# PHL1445 individual LC
    q = 0.087049
    dphi = 0.0538438
    rwd = 0.019544
    ulimb = 0.35
    rdisc = 0.37882
    rexp = 0.34916
    az = 164.16
    frac = 0.13908
    scale = 0.02618
    exp1 = 1.2
    exp2 = 0.9
    tilt = 120.0
    yaw = -10.0
    fwd = 0.12865
    fdisc = 0.04516
    fbs = 0.07855
    fd = 0.0001
    off = -0.000078'''
    
    '''# PHL1445 modified
    q = 0.057049
    dphi = 0.043844
    rwd = 0.013545
    ulimb = 0.35
    rdisc = 0.5788224
    rexp = 0.449158
    az = 114.16167
    frac = 0.139078
    scale = 0.0231764
    exp1 = 2.0
    exp2 = 1.0
    tilt = 100.0
    yaw = 1.0
    fwd = 0.178650
    fdisc = 0.048163
    fbs = 0.0245461
    fd = 0.001
    off = -0.000078024'''
    
    
    '''# PHL1445 after mcmc (averaged, smoothed)
    q = 0.06810
    dphi = 0.05485
    rwd = 0.02221
    ulimb = 0.35
    rdisc = 0.64194
    rexp = 2.0739
    az = 82.198
    frac = 0.3721
    scale = 0.1545
    exp1 = 3.269
    exp2 = 1.3341
    tilt = 55.8135
    yaw = 19.791
    fwd = 0.1281
    fdisc = 0.05726
    fbs = 0.05707
    fd = 0.01741
    off = 0.000021'''
    
    '''# PHL1445 after mcmc (individual, smoothed)
    q = 0.0796
    dphi = 0.0547
    rwd = 0.0206
    ulimb = 0.350
    rdisc = 0.370
    rexp = 0.174
    az = 163
    frac = 0.091
    scale = 0.0246
    exp1 = 2.60
    exp2 = 0.805
    tilt = 69.9
    yaw = 1.57
    fwd = 0.128
    fdisc = 0.059
    fbs = 0.060
    fd = 0.0008
    off = -0.000043'''
    
    '''# SDSS0901
    q = 0.2
    dphi = 0.0574
    rwd = 0.0153
    ulimb = 0.345
    rdisc = 0.455
    rexp = 0.21
    az = 117
    frac = 0.21
    scale = 0.009
    exp1 = 0.11
    exp2 = 0.65
    tilt = 126.0
    yaw = 18
    fwd = 0.0246
    fdisc = 0.0084
    fbs = 0.0251
    fd = 0.0013
    off = -0.0001'''
    
    '''# SDSS0901 modified
    q = 0.18
    dphi = 0.0504
    rwd = 0.023
    ulimb = 0.345
    rdisc = 0.305
    rexp = 0.21
    az = 140
    frac = 0.21
    scale = 0.009
    exp1 = 0.11
    exp2 = 0.65
    tilt = 146.0
    yaw = 18
    fwd = 0.0206
    fdisc = 0.0084
    fbs = 0.0301
    fd = 0.0013
    off = -0.0001'''
    
    '''# ASASSN14ag
    q = 0.100049
    dphi = 0.058844
    rwd = 0.025545
    ulimb = 0.35
    rdisc = 0.5788224
    rexp = 0.449158
    az = 114.16167
    frac = 0.139078
    scale = 0.0251764
    exp1 = 2.0
    exp2 = 1.0
    tilt = 100.0
    yaw = 1.0
    fwd = 0.08850
    fdisc = 0.048163
    fbs = 0.0505461
    fd = 0.001
    off = -0.00078024'''


    #also need hyperparameters for GP
    #amp_gp = np.log( 0.05*y.mean() ) #5% of mean flux
    #tau_gp = np.log( 30./86400. ) #30 secs
    #print tau_gp
    amp_gp = -6.437
    tau_gp = -7.925
    
    guessP = np.array([amp_gp,tau_gp,fwd,fdisc,fbs,fd,q,dphi,rdisc,ulimb,rwd,scale,az,frac,rexp,off, \
                      exp1,exp2,tilt,yaw])

    # is our starting position legal
    if np.isinf( ln_prior_gp(guessP) ):
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
        nwalkers = 100
        nthreads = 6
        nburn = 5000
        nprod = 5000
        mcmcPars = (nwalkers,nburn,nburn,nprod,nthreads)
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
    
