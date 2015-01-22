import numpy as np
import matplotlib.pyplot as plt
from trm import roche
import sys
import lfit
import emcee
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
    tol = 1.0e-3
    maxphi = roche.findphi(pars[4],90.0) #dphi when i is slightly less than 90
    prior = Prior('uniform',0.001,maxphi-tol)
    lnp += prior.ln_prob(pars[5])

    #Disc radius (XL1) 
    prior = Prior('uniform',0.35,0.9)
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

def ln_likelihood(pars,phi,width,y,e,cv):
    yfit = model(pars,phi,width,cv)
    return -0.5*(np.sum( np.log( 2.0*np.pi*e**2 ) ) + chisq(y,yfit,e))

def ln_prob(pars,phi,width,y,e,cv):
    lnp = ln_prior(pars)
    if np.isfinite(lnp):
        return lnp + ln_likelihood(pars,phi,width,y,e,cv)
    else:
        return lnp
    

if __name__ == "__main__":

    
    #Input lightcurve data from txt file
    import argparse
    parser = argparse.ArgumentParser(description='Fit CV lightcurves with lfit')
    parser.add_argument('file',action='store',help='input file (x,y,e)')
    parser.add_argument('-f','--fit',action='store_true',help='actually fit, otherwise just plot')
    args = parser.parse_args()
    file = args.file
    toFit = args.fit

    thisFile = aBug
    
    x,y,e = np.loadtxt(file,skiprows=16).T
    width = np.mean(np.diff(x))*np.ones_like(x)/2.
    
    q = 0.14248
    dphi = 0.064003
    rwd = 0.008196
    ulimb = 0.346000
    rdisc = 0.423139
    rexp = 1.725406
    az = 141.24466
    frac = 0.244912
    scale = 0.038624
    exp1 = 2.0
    exp2 = 1.0
    tilt = 60.0
    yaw = 1.0
    fwd = 0.0124172
    fdisc = 0.0080908
    fbs = 0.0266768
    fd = 0.00111947
    off = 0.000145788

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
        nthreads = 8
        p0 = emcee.utils.sample_ball(guessP,0.05*guessP,size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[x,width,y,e,myCV])

        #Burn-in
        nburn = 1000
        pos, prob, state = run_burnin(sampler,p0,nburn)

    
        #Production
        sampler.reset()
        nprod = 1000
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

    #Plot model & data
    xf = np.linspace(x.min(),x.max(),1000)
    wf = 0.5*np.mean(np.diff(xf))*np.ones_like(xf)
    
    yf = model(bestPars,xf,wf,myCV)
    plt.plot(xf,yf,'r-')
    plt.errorbar(x,y,yerr=e,fmt='.',color='k',capsize=0)
    plt.xlabel('Orbital Phase')
    plt.ylabel('Flux')
    plt.savefig('bestFit.pdf')
    plt.xlim(-0.1,0.15)
    plt.show()

