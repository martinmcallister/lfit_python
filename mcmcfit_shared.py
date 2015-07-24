import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from trm import roche
import sys
import lfit
import emcee
import warnings
import GaussianProcess as GP
from mcmc_utils import *
import seaborn
from collections import MutableSequence
from model import Model

# parallellise with MPIPool
from emcee.utils import MPIPool

class LCModel(Model):
    '''CV lightcurve model for multiple eclipses
       can be passed to routines for calculating model, chisq, prior prob etc
       
       can add eclipses at will with routing addEcl. All eclipses share q, dphi, rwd, 
       limb darkening. All other parameters vary. You cannot mix and match complex
       and simple bright spot models for eclipses; all must use the same type of
       bright spot.
       
       Also behaves as a list of the current values of all variable parameters - 
       this allows it to be seamlessly used with emcee'''
       
    def __init__(self,parList,complex,nel_disc=1000,nel_donor=400):
        '''parameter list should be a 14 element or 18 element dictionary of Param objects
        in order these are:
        wdFlux, dFlux, sFlux, rsFlux, q, dphi, rdisc, ulimb, rwd, scale, az, fis, dexp, phi0
        and optional pars are
        exp1, exp2, tilt, yaw
        '''
        
        super(LCModel,self).__init__(parList)
        self.complex = complex
        
        if complex:
            assert len(parList)==18, "Wrong number of parameters"
        else:
            assert len(parList)==14, "Wrong number of parameters"
        
        # we actually need an LFIT CV object to do the calculations
        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
            'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        if complex:
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
        parVals = [self.getValue(name) for name in parNames]
        self.cv = lfit.CV(parVals)
        
        # How many eclipses?
        self.necl = 1

    def addEclipse(self,parList):
        '''parList should be a list of 11 or 15 Param objects, depending on the complexity
        of the bright spot model. In turn these should be
        wdFlux, dFlux, sFlux, rsFlux, rdisc, ulimb, scale, az, fis, dexp, phi0
        and optional pars are
        exp1, exp2, tilt, yaw
        '''    
        if self.complex:
            assert len(parList) == 15, "Wrong number of parameters"
        else:
            assert len(parList) == 11, "Wrong number of parameters"
            
        self.necl += 1
        self.plist.extend(parList)
        
    def calc(self,ecl,phi,width=None):
        '''we have to extract the current value of the parameters for this ecl, and 
        calculate the CV flux'''

        # parNameTemplate needs updating depending on eclipse number
        # for loop does this
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}', 'q', 'dphi',\
            'rdisc_{0}', 'ulimb_{0}', 'rwd', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']
        if complex:
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])
        
        parNames = [template.format(ecl) for template in parNameTemplate]
            
        parVals = [self.getValue(name) for name in parNames]
        return self.cv.calcFlux(parVals,phi,width)
        
    def chisq(self,phi,y,e,width=None):
        retVal = 0.0
        for iecl in range(self.necl):
            if width:
                thisWidth=width[iecl]
            else:
                thisWidth=None
            resids = ( y[iecl] - self.calc(iecl,phi[iecl],thisWidth) ) / e[iecl]
            retVal += np.sum(resids**2)
        return retVal
        
    def ln_prior(self):
        retVal = super(LCModel,self).ln_prior()

        # add in special cases here
        # dphi
        tol = 1.0e-6
        try:
            q = self.getParam('q')
            dphi = self.getParam('dphi')
            maxphi = roche.findphi(q.currVal,90.0) #dphi when i is slightly less than 90
            if dphi.currVal > maxphi-tol:
                retVal += -np.inf
            else:
                retVal += dphi.prior.ln_prob(dphi.currVal)
        except:
            # we get here when roche.findphi raises error - usually invalid q
            retVal += -np.inf
        
        #Disc radius (XL1) 
        try:
            xl1 = roche.xl1(q.currVal) # xl1/a
            maxrdisc = 0.46/xl1 # maximum size disc can be without precessing
            rdiscTemplate = 'rdisc_{0}'
            for iecl in range(self.necl):
                rdisc = self.getParam(rdiscTemplate.format(iecl))
                if rdisc.currVal > maxrdisc:
                    retVal += -np.inf
                else:
                    retVal += rdisc.prior.ln_prob(rdisc.currVal)
        except:
            # we get here when roche.findphi raises error - usually invalid q
            retVal += -np.inf
        
        #BS scale (XL1)
        rwd = self.getParam('rwd')
        minscale = rwd.currVal/3
        maxscale = rwd.currVal*3
        scaleTemplate = 'scale_{0}'
        for iecl in range(self.necl):
            scale = self.getParam(scaleTemplate.format(iecl))
            if scale.currVal < minscale or scale.currVal > maxscale:
                retVal += -np.inf
            else:
                retVal += scale.prior.ln_prob(scale.currVal)
            
        #BS az
        slop = 40.0
        try:
            # find position of bright spot where it hits disc
            # will fail if q invalid
            azTemplate = 'az_{0}'
            for iecl in range(self.necl):
                rdisc = self.getParam(rdiscTemplate.format(iecl))
                rd_a = rdisc.currVal*xl1
                az = self.getParam(azTemplate.format(iecl))
                # Does stream miss disc? (disc/a < 0.2 or > 0.65 )
                # if so, Tom's code will fail
                x,y,vx,vy = roche.bspot(q.currVal,rd_a)
                # find tangent to disc at this point
                alpha = np.degrees(np.arctan2(y,x))
                # alpha is between -90 and 90. if negative spot lags disc ie alpha > 90
                if alpha < 0: alpha = 90-alpha
                tangent = alpha + 90 # disc tangent
                minaz = max(0,tangent-slop)
                maxaz = min(178,tangent+slop)
                if az.currVal < minaz or az.currVal > maxaz:
                    retVal += -np.inf
                else:
                    retVal += az.prior.ln_prob(az.currVal)
        except:
            # we get here when roche.findphi raises error - usually invalid q
            retVal += -np.inf
            
        return retVal
         
    def ln_like(self,phi,y,e,width=None):
        lnlike = 0.0
        for iecl in range(self.necl):
            lnlike += np.sum( np.log (2.0*np.pi*e[iecl]**2) )
        return -0.5*(lnlike + self.chisq(phi,y,e,width))
        
    def ln_prob(self,parList,phi,y,e,width=None):
        # update the model to reflect the passed parameters
        self.pars = parList
        lnp = self.ln_prior()
        if np.isfinite(lnp):
            try:
                return lnp + self.ln_like(phi,y,e,width)
            except:
                return -np.inf
        else:
            return lnp
            
class GPLCModel(LCModel):
    def __init__(self,parList,complex,amp_gp,tau_gp,nel_disc=1000,nel_donor=400):
        super(GPLCModel,self).__init__(parList,complex,nel_disc,nel_donor)
        # make sure GP params are variable
        self.plist.append(amp_gp)
        self.plist.append(tau_gp)
        
    def ln_prior_gp(self):
        retVal=0.0
        priors_pars_shared = ['amp_gp','tau_gp']
        for par in priors_pars_shared:
            param = getattr(self,par)
            if param.isVar:
                retVal += param.prior.ln_prob(param.currVal)
        return retVal
        
    def ln_prior(self):
        return self.ln_prior_base() + self.ln_prior_gp()    
        
    def createGP(self,parList,phi):
        # check this, does it change if complex?
        a, tau = np.exp(parList[:2])
        dphi, phiOff = parList[7],parList[15]
        
        k_out = a*GP.Matern32Kernel(tau)
        k_in    = 0.01*a*GP.Matern32Kernel(tau)
        
        # Find location of all changepoints
        changepoints = []
        for n in range (int(phi[1]),int(phi[-1])+1,1):
            changepoints.append(n-dphi/2.)
            changepoints.append(n+dphi/2.)  

        # Depending on number of changepoints, create kernel structure
        kernel_struc = [k_out]      
        for k in range (int(phi[1]),int(phi[-1])+1,1):
            kernel_struc.append(k_in)
            kernel_struc.append(k_out)
        
        # create kernel with changepoints 
        # obviously need one more kernel than changepoints!
        kernel = GP.DrasticChangepointKernel(kernel_struc,changepoints)
        
        # create GPs using this kernel
        gp = GP.GaussianProcess(kernel)
        return gp
        
    def ln_like(self,parList,phi,y,e,width=None):
        # new ln_like function, using GPs, looping over each eclipse
        lnlike = 0.0
        
        for iecl in range(self.necl):
            gp = self.createGP(parList,phi[iecl])
            gp.compute(phi[iecl],e[iecl])
            # calculate the model
            if width:
                thisWidth=width[iecl]
            else:
                thisWidth=None
            resids = y[iecl] - self.calc(iecl,phi[iecl],thisWidth)
                                
            # check for bugs in model
            if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
                print parList
                print warning.warn('model gave nan or inf answers')
                return -np.inf
                                
            # now calculate ln_like
                                
            lnlike += gp.lnlikelihood(resids)         
        return lnlike
            
def parseInput(file):
        blob = np.loadtxt(file,dtype='string',delimiter='\n')
        input_dict = {}
        for line in blob:
                k,v = line.split('=')
                input_dict[k.strip()] = v.strip()
        return input_dict
            
if __name__ == "__main__":

    #Input lightcurve data from txt file
    import argparse
    parser = argparse.ArgumentParser(description='Fit CV lightcurves with lfit')
    parser.add_argument('file',action='store',help='input file')
    args = parser.parse_args()

    input_dict = parseInput(args.file)  

    nburn    = int( input_dict['nburn'] )
    nprod    = int( input_dict['nprod'] )
    nthreads = int( input_dict['nthread'] )
    nwalkers = int( input_dict['nwalkers'] )
    scatter  = float( input_dict['scatter'] )
    toFit    = int( input_dict['fit'] )
    
    neclipses = int( input_dict['neclipses'] )
    complex    = bool( int(input_dict['complex']) )
    useGP      = bool( int(input_dict['useGP']) )
    amp_gp = Param.fromString('amp_gp', input_dict['amp_gp'])
    tau_gp = Param.fromString('tau_gp', input_dict['tau_gp'])
    
    files = []
    output_files = []
    for ecl in range(1,1+neclipses):
        files.append( input_dict['file_{0}' .format(ecl) ] )
        output_files.append( input_dict['out_{0}' .format(ecl) ] )


    # OUTPUT FILE CODE
    # setup header for output file format
    outfile_header = """#This file contains the data and best fit. 
    #The first three columns are the data (x, y and y error)
    #The next column is the CV flux
    #The next columns are the flux from wd, bright spot, disc and donor
    """
    # create a model from the first eclipses parameters
    parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
        'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
    if complex:
        parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
    parList = [Param.fromString(name, input_dict[name]) for name in parNames]
    
    if useGP:
        model = GPLCModel(parList,complex,amp_gp,tau_gp)
    else:
        model = LCModel(parList,complex)
        
    # then add in additional eclipses as necessary
    parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
        'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']
    if complex:
        parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}']) 
    for ecl in range(1,neclipses):
        parNames = [template.format(ecl) for template in parNameTemplate]
        parList = [Param.fromString(name, input_dict[name]) for name in parNames]
        model.addEclipse(parList)
        
    # store your data in python lists, so that x[0] are the times for eclipse 0, etc.
    x = []
    y = []
    e = []
    w = []
    
    # crop to range if required
    start = float( input_dict['phi_start'] )
    end = float( input_dict['phi_end'] )
    for file in files:
        xt,yt,et = np.loadtxt(file,skiprows=16).T
        wt = np.mean(np.diff(xt))*np.ones_like(xt)/2.
        #xt,wt,yt,et,_ = np.loadtxt(file).T
        mask = (xt>start)&(xt<end)
        x.append(xt[mask])
        y.append(yt[mask])
        e.append(et[mask])
        w.append(wt[mask])
        
    
    # is our starting position legal?
    if np.isinf( model.ln_prior() ):
        print 'Error: starting position violates priors'
        sys.exit(-1)
        
    npars = model.npars
    params = [par.currVal for par in model.pars]

    if toFit:
        # Initialize the MPI-based pool used for parallelization.
        # MPI not used as found to not be quicker
        '''
        pool = MPIPool()

        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
        '''    
        p0 = np.array(params)
        print "initial ln probability = %.2f" % model.ln_prob(p0,x,y,e,w)
        p0 = emcee.utils.sample_ball(p0,scatter*p0,size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,npars,model.ln_prob,args=[x,y,e,w],threads=nthreads)

        #Burn-in
        print 'starting burn-in'
        pos, prob, state = run_burnin(sampler,p0,nburn)

        # run second burn-in scattered around best fit of previous burnin
        # DFM reports this can help convergence in difficult cases
        print 'starting second burn-in'
        p0 = pos[np.argmax(prob)]
        p0 = emcee.utils.sample_ball(p0,scatter*p0,size=nwalkers)
        pos, prob, state = run_burnin(sampler,p0,nburn)

        #Production
        sampler.reset()
        print 'starting main mcmc chain'
        sampler = run_mcmc_save(sampler,pos,nprod,state,"chain2.txt")  
        
        # stop parallelism
        #pool.close()
        
        chain = flatchain(sampler.chain,npars,thin=10)
        
        # Print out parameters
        params = []
        for i in range(npars):
            par = chain[:,i]
            lolim,best,uplim = np.percentile(par,[16,50,84])
            print "%s = %f +%f -%f" % (model.lookuptable[i],best,uplim-best,best-lolim)
            params.append(best)
        # update model with best params
        model.pars = params
                         
    print '\nFor this model:\n'
    dataSize = np.sum((xa.size for xa in x))
    print "Chisq          = %.2f (%d D.O.F)" % (model.chisq(x,y,e,w),dataSize - model.npars - 1)
    print "ln prior       = %.2f" % model.ln_prior()
    print "ln likelihood = %.2f" % model.ln_like(x,y,e,w)
    print "ln probability = %.2f" % model.ln_prob(params,x,y,e,w)
    
    # Plot model & data
    gs = gridspec.GridSpec(2,neclipses,height_ratios=[2,1])
    gs.update(hspace=0.0)
    seaborn.set()

    LHplot = True
    for iecl in range(neclipses):
        xp = x[iecl]
        yp = y[iecl]
        ep = e[iecl]
        wp = w[iecl]
           
        xf = np.linspace(xp.min(),xp.max(),1000)
        wf = 0.5*np.mean(np.diff(xf))*np.ones_like(xf)
        yp_fit = model.calc(iecl,xp,wp)
        yf = model.calc(iecl,xf,wf)

        ax1 = plt.subplot(gs[0,iecl])
        
        # CV model
        ax1.plot(xf,yf)
        ax1.plot(xf,model.cv.yrs)
        ax1.plot(xf,model.cv.ys)
        ax1.plot(xf,model.cv.ywd)
        ax1.plot(xf,model.cv.yd)
        # data
        
        ax1.errorbar(xp,yp,yerr=ep,fmt='.',color='k',capsize=0,alpha=0.5)

        ax2 = plt.subplot(gs[1,iecl],sharex=ax1)
        ax2.errorbar(xp,yp-yp_fit,yerr=ep,color='k',fmt='.',capsize=0,alpha=0.5)
        #ax2.set_xlim(ax1.get_xlim())
        #ax2.set_xlim(-0.1,0.15)

        #labels
        if LHplot:
               ax1.set_ylabel('Flux (mJy)')
               ax2.set_ylabel('Residuals (mJy)')
               LHplot = False
        ax2.set_xlabel('Orbital Phase')
        ax2.yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        
    for ax in plt.gcf().get_axes()[::2]:
        ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
        
        
    plt.savefig('bestFit.pdf')
    #plt.xlim(-0.1,0.15)
    plt.show()
     
