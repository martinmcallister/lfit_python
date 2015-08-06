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
    """CV lightcurve model for multiple eclipses.
    
       Can be passed to routines for calculating model, chisq, prior, prob, etc.
       Can add eclipses at will with addEcl function. All eclipses share q, dphi, rwd.
       All other parameters vary. You cannot mix and match complex and simple bright 
       spot models for eclipses; all must use the same type of bright spot."""
       
    def __init__(self,parList,complex,nel_disc=1000,nel_donor=400):
        """Initialise model.
        
        Parameter list should be a 14 element (non-complex BS) or 18 element (complex BS)
        dictionary of Param objects. These are:
        wdFlux, dFlux, sFlux, rsFlux, q, dphi, rdisc, ulimb, rwd, scale, az, fis, dexp, phi0
        And additional params: exp1, exp2, tilt, yaw"""
        
        # Use of the super function allows abstract class in model.py to be referenced
        # Here the initialise function is referenced
        super(LCModel,self).__init__(parList)
        self.complex = complex
        
        # Need a way of checking number of parameters is correct
        if complex:
            assert len(parList)==18, "Wrong number of parameters"
        else:
            assert len(parList)==14, "Wrong number of parameters"
        
        # We need an LFIT CV object to do the calculations
        # First we create list of parameter names (first eclipse = 0)
        # Then we get values for each parameter through using getValue function from model.py
        # Finally, CV object calculated from these values
        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
            'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        if complex:
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
        parVals = [self.getValue(name) for name in parNames]
        self.cv = lfit.CV(parVals)
        
        # How many eclipses?
        self.necl = 1

    def addEclipse(self,parList):
        """Allows additional eclipses to be added.
        
        Parameter list should include 11 or 15 Param objects (all params individual
        to each eclipse), depending on complexity of the bright spot model. These should be:
        wdFlux, dFlux, sFlux, rsFlux, rdisc, ulimb, scale, az, fis, dexp, phi0
        and additional params: exp1, exp2, tilt, yaw""" 
        
        # Need a way of checking number of parameters is correct  
        if self.complex:
            assert len(parList) == 15, "Wrong number of parameters"
        else:
            assert len(parList) == 11, "Wrong number of parameters"
        
        # How many eclipses?
        self.necl += 1
        # Add params from additional eclipses to existing parameter list
        self.plist.extend(parList)
        
    def calc(self,ecl,phi,width=None):
        """Extracts current parameter values for each eclipse and calculates CV flux."""

        # Template required for parameter names
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}', 'q', 'dphi',\
            'rdisc_{0}', 'ulimb_{0}', 'rwd', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']
        if complex:
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])
        # Template needs updating depending on eclipse number
        parNames = [template.format(ecl) for template in parNameTemplate]
        # List filled in with current parameter values 
        parVals = [self.getValue(name) for name in parNames]
        # CV Flux calculated from list of current parameter values
        return self.cv.calcFlux(parVals,phi,width)
        
    def chisq(self,phi,y,e,width=None):
        """Calculates chisq, which is required in ln_like"""
        retVal = 0.0
        for iecl in range(self.necl):
            if width:
                thisWidth=width[iecl]
            else:
                thisWidth=None
            # chisq calculation
            resids = (y[iecl] - self.calc(iecl,phi[iecl],thisWidth)) / e[iecl]
            retVal += np.sum(resids**2)
        return retVal
        
    def ln_prior(self):
        """Returns the natural log of the prior probability of this model.
        
        Certain parameters (dphi, rdisc, scale, az) need to be treated as special cases,
        as the model contains more prior information than included in the parameter priors"""
        
        # Use of the super function allows abstract class in model.py to be referenced
        # Here the ln_prior function is referenced
        retVal = super(LCModel,self).ln_prior()

        # Remaining part of this function deals with special cases
        # dphi
        tol = 1.0e-6
        try:
            # Uses getParam function from model.py to get the objects of variable parameters
            q = self.getParam('q')
            dphi = self.getParam('dphi')
            # maxphi is dphi when i = 90
            maxphi = roche.findphi(q.currVal,90.0)
            # dphi cannot be greater than (or within a certain tolerance of) maxphi
            if dphi.currVal > maxphi-tol:
                retVal += -np.inf
            else:
                retVal += dphi.prior.ln_prob(dphi.currVal)
        except:
            # We get here when roche.findphi raises error - usually invalid q
            retVal += -np.inf
        
        # rdisc 
        try:
            xl1 = roche.xl1(q.currVal) # xl1/a
            maxrdisc = 0.46/xl1 # Maximum size disc can reach before precessing
            # rdisc is unique to each eclipse, so have to use slightly different method to 
            # obtain its object, compared to q and dphi which are shared parameters
            rdiscTemplate = 'rdisc_{0}'
            for iecl in range(self.necl):
                rdisc = self.getParam(rdiscTemplate.format(iecl))
                # rdisc cannot be greater than maxrdisc
                if rdisc.currVal > maxrdisc:
                    retVal += -np.inf
                else:
                    retVal += rdisc.prior.ln_prob(rdisc.currVal)
        except:
            # We get here when roche.findphi raises error - usually invalid q
            retVal += -np.inf
        
        #BS scale
        rwd = self.getParam('rwd')
        minscale = rwd.currVal/3 # Minimum BS scale equal to 1/3 of rwd
        maxscale = rwd.currVal*3 # Maximum BS scale equal to 3x rwd
        scaleTemplate = 'scale_{0}'
        for iecl in range(self.necl):
            scale = self.getParam(scaleTemplate.format(iecl))
            # BS scale must be within allowed range 
            if scale.currVal < minscale or scale.currVal > maxscale:
                retVal += -np.inf
            else:
                retVal += scale.prior.ln_prob(scale.currVal)
            
        #BS az
        slope = 40.0
        try:
            # Find position of bright spot where it hits disc
            azTemplate = 'az_{0}'
            for iecl in range(self.necl):
                rdisc = self.getParam(rdiscTemplate.format(iecl))
                rd_a = rdisc.currVal*xl1 # rdisc/a
                az = self.getParam(azTemplate.format(iecl))
                # Does stream miss disc? (rdisc/a < 0.2 or rdisc/a > 0.65 )
                # If yes, Tom's code will fail
                # Calculate position of BS
                x,y,vx,vy = roche.bspot(q.currVal,rd_a)
                # Find tangent to disc at this point
                alpha = np.degrees(np.arctan2(y,x))
                # Alpha is between -90 and 90. If negative, spot lags disc (i.e. alpha > 90)
                if alpha < 0: alpha = 90 - alpha
                tangent = alpha + 90 # Disc tangent
                # Calculate minimum and maximum az values using tangent and slope
                minaz = max(0,tangent-slope)
                maxaz = min(178,tangent+slope)
                # BS az must be within allowed range
                if az.currVal < minaz or az.currVal > maxaz:
                    retVal += -np.inf
                else:
                    retVal += az.prior.ln_prob(az.currVal)
        except:
            # We get here when roche.findphi raises error - usually invalid q
            retVal += -np.inf
            
        return retVal
         
    def ln_like(self,phi,y,e,width=None):
        """Calculates the natural log of the likelihood"""
        lnlike = 0.0
        for iecl in range(self.necl):
            if width:
            	thisWidth=width[iecl]
            else:
                thisWidth=None
            resids = y[iecl] - self.calc(iecl,phi[iecl],thisWidth)
            # Check for bugs in model
            if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
                print warning.warn('model gave nan or inf answers')
                return -np.inf
            lnlike += np.sum(np.log(2.0*np.pi*e[iecl]**2))
        return -0.5*(lnlike + self.chisq(phi,y,e,width))
        
    def ln_prob(self,parList,phi,y,e,width=None):
        """Calculates the natural log of the posterior probability (ln_prior + ln_like)"""
        # The model is updated to reflect the passed parameters
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
    """CV lightcurve model for multiple eclipses, with added Gaussian process fitting"""
    
    def __init__(self,parList,complex,amp_gp,tau_gp,nel_disc=1000,nel_donor=400):
        """Initialise model.
        
        Parameter list should be a 16 element (non-complex BS) or 20 element (complex BS)
        dictionary of Param objects. These are:
        amp_gp, tau_gp, wdFlux, dFlux, sFlux, rsFlux, q, dphi, rdisc, ulimb, rwd, scale, az, fis, dexp, phi0
        And additional params: exp1, exp2, tilt, yaw"""
        
        super(GPLCModel,self).__init__(parList,complex,nel_disc,nel_donor)
        # Make sure GP parameters are variable when using this model
        self.plist.append(amp_gp)
        self.plist.append(tau_gp)
        
    def createGP(self,phi):
        """Constructs a kernel, which is used to create Gaussian processes.
        
        Uses values for the two hyperparameters (amp,tau) and dphi, this function: creates
        kernels for both inside and out of eclipse, works out the location of any changepoints
        present, constructs a single (mixed) kernel and uses this kernel to create GPs"""
    
        # Get objects for amp_gp, tau_gp and find the exponential of their current values
        ln_amp = self.getParam('amp_gp')
        ln_tau = self.getParam('tau_gp')
        amp = np.exp(ln_amp.currVal)
        tau = np.exp(ln_tau.currVal)
        # Also get object for dphi, q and rwd as this is required to determine changepoints
        dphi = self.getParam('dphi')
        q = self.getParam('q')
        rwd = self.getParam('rwd')
        
        # Calculate inclination
        inc = roche.findi(q.currVal,dphi.currVal)
        # Calculate wd contact phases 3 and 4
        phi3, phi4 = roche.wdphases(q.currVal, inc, rwd.currVal, ntheta=10)
        # Calculate length of wd egress
        dpwd = phi4 - phi3
        # Distance from changepoints to mideclipse
        dist_cp = 1.25*(dphi.currVal/2.+dpwd/2.)
        
        # Calculate kernels for both out of and in eclipse WD eclipse
        # Kernel inside of WD has much smaller amplitude than that of outside eclipse
        k_out = amp*GP.Matern32Kernel(tau)
        k_in    = 0.01*amp*GP.Matern32Kernel(tau)
        
        # Find location of all changepoints
        changepoints = []
        # the following range construction gives a list
        # of all mid-eclipse phases within phi array
        for n in range (int( phi.min() ), int( phi.max() )+1, 1):
            changepoints.append(n-dist_cp)
            changepoints.append(n+dist_cp)  

        # Depending on number of changepoints, create kernel structure
        kernel_struc = [k_out]      
        for k in range (int( phi.min() ), int( phi.max() )+1, 1):
            kernel_struc.append(k_in)
            kernel_struc.append(k_out)
        
        # Create kernel with changepoints 
        kernel = GP.DrasticChangepointKernel(kernel_struc,changepoints)
        
        # Create GPs using this kernel
        gp = GP.GaussianProcess(kernel)
        return gp
            
    def ln_like(self,phi,y,e,width=None):
        """Calculates the natural log of the likelihood.
        
        This alternative ln_like function uses the createGP function to create Gaussian
        processes"""
        lnlike = 0.0
        # For each eclipse, create (and compute) Gaussian process and calculate the model
        for iecl in range(self.necl):
            gp = self.createGP(phi[iecl])
            gp.compute(phi[iecl],e[iecl])
            if width:
                thisWidth=width[iecl]
            else:
                thisWidth=None
            resids = y[iecl] - self.calc(iecl,phi[iecl],thisWidth)
                                
            # Check for bugs in model
            if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
                print warning.warn('model gave nan or inf answers')
                return -np.inf
                                
            # Calculate ln_like using lnlikelihood function from GaussianProcess.py             
            lnlike += gp.lnlikelihood(resids)         
        return lnlike
                
def parseInput(file):
    """Splits input file up making it easier to read"""
    # Reads in input file and splits it into lines
    blob = np.loadtxt(file,dtype='string',delimiter='\n')
    input_dict = {}
    for line in blob: 
        # Each line is then split at the equals sign
        k,v = line.split('=')
        input_dict[k.strip()] = v.strip()
    return input_dict
            
if __name__ == "__main__":

    # Allows input file to be passed to code from argument line
    import argparse
    parser = argparse.ArgumentParser(description='Fit CV lightcurves with lfit')
    parser.add_argument('file',action='store',help='input file')
    args = parser.parse_args()
    
    # Use parseInput function to read data from input file
    input_dict = parseInput(args.file)  

    # Read in information about mcmc, neclipses, use of complex/GP etc.
    nburn     = int(input_dict['nburn'])
    nprod     = int(input_dict['nprod'])
    nthreads  = int(input_dict['nthread'])
    nwalkers  = int(input_dict['nwalkers'])
    scatter   = float(input_dict['scatter'])
    toFit     = int(input_dict['fit'])
    neclipses = int(input_dict['neclipses'])
    complex   = bool(int(input_dict['complex']))
    useGP     = bool(int(input_dict['useGP']))
    
    # Read in GP params using fromString function from mcmc_utils.py
    amp_gp = Param.fromString('amp_gp', input_dict['amp_gp'])
    tau_gp = Param.fromString('tau_gp', input_dict['tau_gp'])
    
    # Read in file names containing eclipse data, as well as output file names
    files = []
    output_files = []
    for ecl in range(0,neclipses):
        files.append(input_dict['file_{0}'.format(ecl)])
        output_files.append(input_dict['out_{0}'.format(ecl)])


    # Output file code - needs completing
    # This file contains the data and best fit. 
    # The first three columns are the data (x, y and y error)
    # The next column is the CV flux
    # The next columns are the flux from wd, bright spot, disc and donor
    # Setup header for output file format
    # outfile_header = 

    # Create a model from the first eclipses (eclipse 0) parameters
    parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
        'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
    if complex:
        parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
    # List of values obtained from input file using fromString function from mcmc_utils.py
    parList = [Param.fromString(name, input_dict[name]) for name in parNames]
    
    # If fitting using GPs use GPLCModel, else use LCModel
    if useGP:
        model = GPLCModel(parList,complex,amp_gp,tau_gp)
    else:
        model = LCModel(parList,complex)
        
    # pickle is used for parallelisation
    # pickle cannot pickle methods of classes, so we wrap
    # the ln_prob function here to make something that can be pickled
    def ln_prob(parList,phi,y,e,width=None):
        return model.ln_prob(parList,phi,y,e,width=None)
    
    # Add in additional eclipses as necessary
    parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
        'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']
    if complex:
        parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}']) 
    for ecl in range(1,neclipses):
        # This line changes the eclipse number for each parameter name
        parNames = [template.format(ecl) for template in parNameTemplate]
        # List of values obtained from input file using fromString function from mcmc_utils.py
        parList = [Param.fromString(name, input_dict[name]) for name in parNames]
        # Use addEclipse function defined above to add eclipse parameters to parameter list
        model.addEclipse(parList)
        
    # Store your data in python lists, so that x[0] are the times for eclipse 0, etc.
    x = []
    y = []
    e = []
    w = []
    
    # Crop to range given in input file
    start = float(input_dict['phi_start'])
    end = float(input_dict['phi_end'])
    # Read in eclipse data
    for file in files:
        xt,yt,et = np.loadtxt(file,skiprows=16).T
        wt = np.mean(np.diff(xt))*np.ones_like(xt)/2.
        # Create mask
        mask = (xt > start) & (xt < end)
        x.append(xt[mask])
        y.append(yt[mask])
        e.append(et[mask])
        w.append(wt[mask])
        
    
    # Is starting position legal?
    if np.isinf(model.ln_prior()):
        print 'Error: starting position violates priors'
        sys.exit(-1)
      
    # How many parameters?  
    npars = model.npars
    # Current values of all parameters
    params = [par.currVal for par in model.pars]
    
    # The following code will only run if option to fit has been selected in input file
    if toFit:
        # Initialize the MPI-based pool used for parallelization.
        # Need to look into this
        '''
        pool = MPIPool()

        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
        '''    
        p0 = np.array(params) # Starting parameters
        
        '''
        BIZARRO WORLD!
        Calling the ln_prob function once outside of multiprocessing
        causes multiprocessing calls to the same function to hang or segfault
        when using numpy/scipy on OS X. This is a known bug when using mp
        in combination with the BLAS library (cho_factor uses this).
        
        http://stackoverflow.com/questions/19705200/multiprocessing-with-numpy-makes-python-quit-unexpectedly-on-osx
        '''
        # print "initial ln probability = %.2f" % model.ln_prob(p0,x,y,e,w)
        
        # Produce a ball of walkers around p0
        p0 = emcee.utils.sample_ball(p0,scatter*p0,size=nwalkers)
        
        '''
        print 'probabilities of walker positions: '
        for i, par in enumerate(p0):
            print '%d = %.2f' % (i,model.ln_prob(par,x,y,e,w))
        '''
        
        # Instantiate Ensemble sampler
        sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[x,y,e,w],threads=nthreads)

        # Burn-in
        print 'starting burn-in'
        # Run burn-in stage of mcmc using run_burnin function from mcmc_utils.py
        pos, prob, state = run_burnin(sampler,p0,nburn)

        # Run second burn-in stage, scattered around best fit of previous burn-in
        # DFM (emcee creator) reports this can help convergence in difficult cases
        print 'starting second burn-in'
        p0 = pos[np.argmax(prob)]
        p0 = emcee.utils.sample_ball(p0,scatter*p0,size=nwalkers)
        pos, prob, state = run_burnin(sampler,p0,nburn)

        #Production
        sampler.reset()
        print 'starting main mcmc chain'
        # Run production stage of mcmc using run_mcmc_save function from mcmc_utils.py
        sampler = run_mcmc_save(sampler,pos,nprod,state,"chain2.txt")  
        '''
        stop parallelism
        pool.close()
        '''
        # Create a chain (i.e. collect results from all walkers) using flatchain function
        # from mcmc_utils.py
        chain = flatchain(sampler.chain,npars,thin=10)
        
        # Print out individual parameters
        params = []
        for i in range(npars):
            par = chain[:,i]
            lolim,best,uplim = np.percentile(par,[16,50,84])
            print "%s = %f +%f -%f" % (model.lookuptable[i],best,uplim-best,best-lolim)
            params.append(best)
        # Update model with best parameters
        model.pars = params
    
    # Print out chisq, ln prior, ln likelihood and ln probability for the model                 
    print '\nFor this model:\n'
    # Size of data required in order to calculate degrees of freedom (D.O.F)
    dataSize = np.sum((xa.size for xa in x))
    print "Chisq          = %.2f (%d D.O.F)" % (model.chisq(x,y,e,w),dataSize - model.npars - 1)
    print "ln prior       = %.2f" % model.ln_prior()
    print "ln likelihood = %.2f" % model.ln_like(x,y,e,w)
    print "ln probability = %.2f" % model.ln_prob(params,x,y,e,w)
    
    # Plot model & data
    # Use of gridspec to help with plotting
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
        res = yp - yp_fit
        
        # Needed for plotting GP 
        if useGP:
            gp = model.createGP(xp)
            gp.compute(xp,ep)
            samples = gp.sample_conditional(res, xp, size = 300)
            mu = np.mean(samples,axis=0)
            std = np.std(samples,axis=0)
            fmu, _ = gp.predict(res, xf)
            
        ax1 = plt.subplot(gs[0,iecl])
        
        # CV model
        ax1.plot(xf,yf)
        ax1.plot(xf,model.cv.yrs)
        ax1.plot(xf,model.cv.ys)
        ax1.plot(xf,model.cv.ywd)
        ax1.plot(xf,model.cv.yd)
        if useGP:
			# Plot GP
        	ax1.plot(xf,yf+fmu,color='r',linestyle='--',alpha=0.75)
        
        # Data
        ax1.errorbar(xp,yp,yerr=ep,fmt='.',color='k',capsize=0,alpha=0.5)
        ax2 = plt.subplot(gs[1,iecl],sharex=ax1)
        ax2.errorbar(xp,yp-yp_fit,yerr=ep,color='k',fmt='.',capsize=0,alpha=0.5)
        #ax2.set_xlim(ax1.get_xlim())
        #ax2.set_xlim(-0.1,0.15)
        if useGP:
        	ax2.fill_between(xp,mu+2.0*std,mu-2.0*std,color='r',alpha=0.4)

        # Labels
        if LHplot:
               ax1.set_ylabel('Flux (mJy)')
               ax2.set_ylabel('Residuals (mJy)')
               LHplot = False
        ax2.set_xlabel('Orbital Phase')
        ax2.yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        
    for ax in plt.gcf().get_axes()[::2]:
        ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
        
    # Save plot image 
    plt.savefig('bestFit.pdf')
    plt.show()
     
