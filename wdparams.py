from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import range
from builtins import object
from past.utils import old_div
import numpy
import emcee
from mcmc_utils import *
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from collections import MutableSequence
import warnings
import sys
import seaborn

class wdModel(MutableSequence):
    '''wd model
    can be passed to MCMC routines for calculating model and chisq, and prior prob
    
    also behaves like a list, of the current values of all parameters 
    this enables it to be seamlessly used with emcee'''
    
    # arguments are Param objects (see mcmc_utils)
    def __init__(self,teff,logg,dist,ebv):
        self.teff = teff
        self.logg = logg
        self.dist = dist
        self.ebv  = ebv
        
        # initialise list bit of object with parameters
        self.data = [self.teff,self.logg,self.dist,self.ebv]
        
    # these routines are needed so object will behave like a list
    def __getitem__(self,ind):
        return self.data[ind].currVal
    def __setitem__(self,ind,val):
        self.data[ind].currVal = val
    def __delitem__(self,ind):
        self.data.remove(ind)
    def __len__(self):
        return len(self.data)
    def insert(self,ind,val):
        self.data.insert(ind,val)
    @property
    def npars(self):
        return len(self.data)    
        
def parseInput(file):
    ''' reads in a file of key = value entries and returns a dictionary'''
    # Reads in input file and splits it into lines
    blob = np.loadtxt(file,dtype='string',delimiter='\n')
    input_dict = {}
    for line in blob:
        # Each line is then split at the equals sign
        k,v = line.split('=')
        input_dict[k.strip()] = v.strip()
    return input_dict
    
def model(thisModel,mask):
    t, g, d, ebv = thisModel
    
    # load bergeron models
    data = np.loadtxt('Bergeron/da_ugrizkg5.txt')

    teffs = np.unique(data[:,0])
    loggs = np.unique(data[:,1])
    
    nteff = len(teffs)
    nlogg = len(loggs)
    assert t <= teffs.max()
    assert t >= teffs.min()
    #assert g >= loggs.min()
    #assert g <= loggs.max()
        
    abs_mags = []
    # u data in col 4, g in 5, r in 6, i in 7, z in 8, kg5 in 9
    for col_indx in range(4,10):
        z = data[:,col_indx]
        z = z.reshape((nlogg,nteff))
        # cubic bivariate spline interpolation
        func = interp.RectBivariateSpline(loggs,teffs,z,kx=3,ky=3)
        abs_mags.append(func(g,t)[0,0])
    abs_mags = np.array(abs_mags)
    
    # A_x/E(B-V) extinction from Cardelli (1989)
    # Where are these values from?? (KG5 estimated)
    
    ext       = ebv*np.array([5.155,3.793,2.751,2.086,1.479,3.5])
    dmod      = 5.0*np.log10(d/10.0)
    app_red_mags = abs_mags + ext + dmod
    
    #return app_red_mags
    return 3631e3*10**(-0.4*app_red_mags[mask])

def ln_prior(thisModel):
    lnp = 0.0

    #teff, (usually uniform between allowed range - 6 to 90,000)
    param = thisModel.teff
    lnp += param.prior.ln_prob(param.currVal)

    #logg, uniform between allowed range (7.01 to 8.99), or Gaussian from constraints
    param = thisModel.logg
    lnp += param.prior.ln_prob(param.currVal)    
    
    # distance, uniform between 50 and 10,000 pc
    # (this is biassed against real distances vs actual prior)
    # so we scale by volume of thin radius step dr (prop. to r**2/50**2)
    param = thisModel.dist
    loLim = param.prior.p1
    val   = param.currVal
    #lnp += (val/loLim)**2 * param.prior.ln_prob(val)
    lnp += param.prior.ln_prob(val)

    # reddening, cannot exceed galactic value of 0.121
    param = thisModel.ebv
    lnp += param.prior.ln_prob(param.currVal)    
    return lnp
    
def chisq(thisModel,y,e,mask):
    m = model(thisModel,mask)
    try:
        resids = old_div((y[mask] - m), e[mask])
        return np.sum(resids*resids)
    except:
        return np.inf
        
def ln_likelihood(thisModel,y,e,mask):
    errs = e[mask]
    return -0.5*(np.sum( np.log( 2.0*np.pi*errs**2 ) ) + chisq(thisModel,y,e,mask))
    
def ln_prob(pars,thisModel,y,e,mask):

    # first we update the model to use the pars suggested by the MCMC chain
    for i in range(thisModel.npars):
        thisModel[i] = pars[i]
    
    # now calculate log prob
    lnp = ln_prior(thisModel)
    if np.isfinite(lnp):
        return lnp + ln_likelihood(thisModel,y,e,mask)
    else:
        return lnp
        
class Flux(object):
    def __init__(self,val,err,band):
        self.val = val
        self.err = err
        self.band = band 
        self.mag = 2.5*numpy.log10(old_div(3631000,self.val))
        self.magerr = 2.5*0.434*(old_div(self.err,self.val))
        
def plotFluxes(fluxes,fluxes_err,mask,model):

    teff = model[0] 
    logg = model[1]
    d = model[2]
    ebv = model[3] 
    
    # load bergeron models
    data = numpy.loadtxt('Bergeron/da_ugrizkg5.txt')
    
    teffs = np.unique(data[:,0])
    loggs = np.unique(data[:,1])
    
    nteff = len(teffs)
    nlogg = len(loggs)

    abs_mags = []
    # u data in col 4, g in 5, r in 6, i in 7, z in 8, kg5 in 9
    for col_indx in range(4,10):
        z = data[:,col_indx]
        z = z.reshape((nlogg,nteff))
        # cubic bivariate spline interpolation
        func = interp.RectBivariateSpline(loggs,teffs,z,kx=3,ky=3)
        abs_mags.append(func(logg,teff)[0,0])
    abs_mags = np.array(abs_mags)
    
    # A_x/E(B-V) extinction from Cardelli (1989)
    # Where are these values from?? (KG5 estimated)
    
    ext       = ebv*np.array([5.155,3.793,2.751,2.086,1.479,3.5])
    dmod      = 5.0*np.log10(old_div(d,10.0))
    app_red_mags = abs_mags + ext + dmod
    
    # calculate fluxes from model magnitudes
    model_fluxes = 3631e3*10**(-0.4*app_red_mags)
    
    # central wavelengths
    wavelengths = np.array([355.7,482.5,626.1,767.2,909.7,507.5])
    
    seaborn.set(style='ticks')
    seaborn.set_style({"xtick.direction": "in","ytick.direction": "in"})
    
    plt.errorbar(wavelengths[mask],model_fluxes[mask],xerr=None,yerr=None,fmt='o',ls='none',color='r',markersize=6,capsize=None)
    plt.errorbar(wavelengths[mask],fluxes[mask],xerr=None,yerr=fluxes_err[mask],fmt='o',ls='none',color='b',markersize=6,linewidth=1,capsize=None)
    plt.xlabel('Wavelength (nm)', fontsize=16)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.ylabel('Flux (mJy)', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(top='on',right='on')
    #plt.xlim(300,700)
    #plt.ylim(0.03,0.05)
    #plt.yticks(np.arange(0.03,0.05,0.005))
    plt.subplots_adjust(bottom=0.10, top=0.98, left=0.11, right=0.975)
    plt.savefig('fluxPlot.pdf')
    plt.show()

def plotColors(mags):

    # load bergeron models
    data = numpy.loadtxt('Bergeron/da_ugrizkg5.txt')

    # bergeron model magnitudes
    umags = data[:,4]
    gmags = data[:,5]
    rmags = data[:,6]
    imags = data[:,7]
    zmags = data[:,8]
    kg5mags = data[:,9]
    
    # calculate colours   
    ug = umags-gmags
    gr = gmags-rmags

    # make grid of teff, logg and colours
    teff = numpy.unique(data[:,0])
    logg = numpy.unique(data[:,1])
    nteff = len(teff)
    nlogg = len(logg)    
    # reshape colours onto 2D grid of (logg, teff)
    ug = ug.reshape((nlogg,nteff))
    gr = gr.reshape((nlogg,nteff))
    
    # DATA!
    # If u band data available, chances are g and r data available too
    # u-g
    col1  = mags[0].mag - mags[1].mag
    col1e = numpy.sqrt(mags[0].magerr**2 + mags[1].magerr**2)
    col1l = mags[0].band + '-' + mags[1].band
    
    if rband_used:
        # g-r 
        col2  = mags[1].mag - mags[2].mag
        col2e = numpy.sqrt(mags[1].magerr**2 + mags[2].magerr**2)
        col2l = mags[1].band + '-' + mags[2].band
        
    else:
        # g-i 
        col2  = mags[1].mag - mags[3].mag
        col2e = numpy.sqrt(mags[1].magerr**2 + mags[3].magerr**2)
        col2l = mags[1].band + '-' + mags[3].band
    
    print('%s = %f +/- %f' % (col1l,col1,col1e))
    print('%s = %f +/- %f' % (col2l,col2,col2e))

    # now plot everthing
    for a in range(len(logg)):
        plt.plot(ug[a,:],gr[a,:],'k-')
        
        
    for a in range(0,len(teff),4):
        plt.plot(ug[:,a],gr[:,a],'r--')
        
    # annotate for log g
    #xa = ug[0,nteff/3]+0.03
    #ya = gr[0,nteff/3]-0.02
    #t = plt.annotate('log g = 7.0',xy=(xa,ya),color='k',horizontalalignment='center', verticalalignment='center',size='small')
    #t.set_rotation(30.0)
    #xa = ug[-1,nteff/3]-0.05
    #ya = gr[-1,nteff/3]+0.0
    #t = plt.annotate('log g = 9.0',xy=(xa,ya),color='k',horizontalalignment='center', verticalalignment='center',size='small')
    #t.set_rotation(45.0)
    
    # annotate for teff
    xa = ug[0,4] + 0.03
    ya = gr[0,4]
    val = teff[4]
    t = plt.annotate('T = %d K' % val,xy=(xa,ya),color='r',horizontalalignment='left', verticalalignment='center',size='small')
    t.set_rotation(0.0)
    xa = ug[0,8] + 0.03
    ya = gr[0,8]
    val = teff[8]
    t = plt.annotate('T = %d K' % val,xy=(xa,ya),color='r',horizontalalignment='left', verticalalignment='center',size='small')
    t.set_rotation(0.0)
    xa = ug[0,20] + 0.01
    ya = gr[0,20] - 0.01
    val = teff[20]
    t = plt.annotate('T = %d K' % val,xy=(xa,ya),color='r',horizontalalignment='left', verticalalignment='top',size='small')
    t.set_rotation(0.0)
    xa = ug[0,24] + 0.01
    ya = gr[0,24] - 0.01
    val = teff[24]
    t = plt.annotate('T = %d K' % val,xy=(xa,ya),color='r',horizontalalignment='left', verticalalignment='top',size='small')
    t.set_rotation(0.0)    
    
    # plot data
    plt.errorbar(col1,col2,xerr=col1e,yerr=col2e,fmt='o',ls='none',color='r',capsize=3)
    plt.xlabel(col1l)
    plt.ylabel(col2l)
    plt.xlim([-0.5,1])
    plt.ylim([-0.5,0.5])
    plt.savefig('colorPlot.pdf')
    plt.show()
        
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    
    # Allows input file to be passed to code from argument line
    import argparse
    parser = argparse.ArgumentParser(description='Fit WD Fluxes')
    parser.add_argument('file',action='store',help="input file")
    
    args = parser.parse_args()
    
    # Use parseInput function to read data from input file
    input_dict = parseInput(args.file)
    
    # Read information about mcmc, priors, neclipses, sys err
    nburn    = int( input_dict['nburn'] )
    nprod    = int( input_dict['nprod'] )
    nthread  = int( input_dict['nthread'] )
    nwalkers = int( input_dict['nwalkers'] )
    scatter  = float( input_dict['scatter'] )
    thin     = int( input_dict['thin'] )
    toFit    = int( input_dict['fit'] )
    teff = Param.fromString('teff', input_dict['teff'] )
    logg = Param.fromString('logg', input_dict['logg'] )
    dist = Param.fromString('dist', input_dict['dist'] )
    ebv = Param.fromString('ebv', input_dict['ebv'] )
    syserr = float( input_dict['syserr'] )
    neclipses = int( input_dict['neclipses'] )
    complex   = int( input_dict['complex'] )
    useGP     = int( input_dict['useGP'] )
    flat      = int( input_dict['flat'] )
    
    # Add in filters used
    filters = []
    for ecl in range(0,neclipses):
        filters.append(input_dict['fil_{0}'.format(ecl)])
    filters = np.array(filters)
    print(filters)
        
    # Load in chain file
    file = input_dict['chain']
    
    if flat:
        fchain = readflatchain(file)
    else:
        #chain = readchain(file)
        chain = readchain_dask(file)
        nwalkers, nsteps, npars = chain.shape
        fchain = flatchain(chain,npars,thin=thin)
    
    # Create array of indexes of same filter type
    uband_filters = np.where(filters == 'u')
    gband_filters = np.where(filters == 'g')
    rband_filters = np.where(filters == 'r')
    iband_filters = np.where(filters == 'i')
    zband_filters = np.where(filters == 'z')
    kg5band_filters = np.where(filters == 'kg5')
    
    # Which filter bands are used?
    uband_used = False
    gband_used = False
    rband_used = False
    iband_used = False
    zband_used = False
    kg5band_used = False
    if len(uband_filters[0] > 0): uband_used = True
    if len(gband_filters[0] > 0): gband_used = True
    if len(rband_filters[0] > 0): rband_used = True
    if len(iband_filters[0] > 0): iband_used = True
    if len(zband_filters[0] > 0): zband_used = True
    if len(kg5band_filters[0] > 0): kg5band_used = True
    
    # Create arrays to be filled with all wd fluxes and mags
    fluxes = [0,0,0,0,0,0]
    fluxes_err = [0,0,0,0,0,0]
    mags = [0,0,0,0,0,0]
    
    if complex == 1:
        a = 15
    else:
        a = 11
        
    if useGP == 1:
        b = 6
    else:
        b = 3
        
    # In some circumstances, the uband eclipse has to be fit separately
    # e.g. when it is of poor quality
    # For this reason, a uband wd flux and error can be input manually
    if uband_used == False:
        while True:
            mode = input('Add seperate u band wd flux and error? (Y/N): ')
            if mode.upper() == 'Y' or mode.upper() == 'N':
                break
            else:
                print("Please answer Y or N ")
            
        if mode.upper() == "Y":
            uflux_in,uflux_err_in = input('Enter uband wd flux and error: ').split()
            uflux_in = float(uflux_in); uflux_err_in = float(uflux_err_in)
            uflux = uflux_in
            uflux_err = np.sqrt(uflux_err_in**2 + (uflux*syserr)**2)
            fluxes[0] = uflux
            fluxes_err[0] = uflux_err
            umag = Flux(uflux,uflux_err,'u')
            mags[0] = umag
            print((uflux,uflux_err))
    
    # For each filter, fill lists with wd fluxes from mcmc chain, then append to main array
    if uband_used:
        uband = []
        uband_filters = uband_filters[0]
        for i in uband_filters:
            if i == 0:
                wdflux = fchain[:,i]
                uband.extend(wdflux)
            else: 
                i = a*i + b
                wdflux = fchain[:,i]
                uband.extend(wdflux)  
                
        uband = np.array(uband)
        # Need to calculate median values and errors
        uflux = np.median(uband)
        uflux_err = np.sqrt((np.std(uband))**2 + (uflux*syserr)**2)
        fluxes[0] = uflux
        fluxes_err[0] = uflux_err
        
        umag = Flux(uflux,uflux_err,'u')
        mags[0] = umag
       
    if gband_used:
        gband = []
        gband_filters = gband_filters[0]
        for i in gband_filters:
            if i == 0:
                wdflux = fchain[:,i]
                gband.extend(wdflux)
            else: 
                i = a*i + b
                wdflux = fchain[:,i]
                gband.extend(wdflux)
      
        gband = np.array(gband)
        gflux = np.median(gband)
        gflux_err = np.sqrt((np.std(gband))**2 + (gflux*syserr)**2)
        fluxes[1] = gflux
        fluxes_err[1] = gflux_err
        
        gmag = Flux(gflux,gflux_err,'g')
        mags[1] = gmag
                
    if rband_used:
        rband = []
        rband_filters = rband_filters[0]
        for i in rband_filters:
            if i == 0:
                wdflux = fchain[:,i]
                rband.extend(wdflux)
            else: 
                i = a*i + b
                wdflux = fchain[:,i]
                rband.extend(wdflux)
                
        rband = np.array(rband)
        rflux = np.median(rband)
        rflux_err = np.sqrt((np.std(rband))**2 + (rflux*syserr)**2)
        fluxes[2] = rflux
        fluxes_err[2] = rflux_err
        
        rmag = Flux(rflux,rflux_err,'r')
        mags[2] = rmag
        
    if iband_used:
        iband = []
        iband_filters = iband_filters[0]
        for i in iband_filters:
            if i == 0:
                wdflux = fchain[:,i]
                iband.extend(wdflux)
            else: 
                i = a*i + b
                wdflux = fchain[:,i]
                iband.extend(wdflux)
        iband = np.array(iband)
        iflux = np.median(iband)
        iflux_err = np.sqrt((np.std(iband)**2 + (iflux*syserr)**2))
        fluxes[3] = iflux
        fluxes_err[3] = iflux_err
                
        imag = Flux(iflux,iflux_err,'i')
        mags[3] = imag
              
    if zband_used:
        zband = []
        zband_filters = zband_filters[0]
        for i in zband_filters:
            if i == 0:
                wdflux = fchain[:,i]
                zband.extend(wdflux)
            else: 
                i = a*i + b
                wdflux = fchain[:,i]
                zband.extend(wdflux)
                
        zband = np.array(zband)
        zflux = np.median(zband)
        zflux_err = np.sqrt((np.std(zband))**2 + (zflux*syserr)**2)
        fluxes[4] = zflux
        fluxes_err[4] = zflux_err
        
        zmag = Flux(zflux,zflux_err,'z')
        mags[4] = zmag 
          
    if kg5band_used:
        kg5band = []
        kg5band_filters = kg5band_filters[0]
        for i in kg5band_filters:
            if i == 0:
                wdflux = fchain[:,i]
                kg5band.extend(wdflux)
            else: 
                i = a*i + b
                wdflux = fchain[:,i]
                kg5band.extend(wdflux)
                
        kg5band = np.array(kg5band)
        kg5flux = np.median(kg5band)
        kg5flux_err = np.sqrt((np.std(kg5band))**2 + (kg5flux*syserr)**2)
        fluxes[5] = kg5flux
        fluxes_err[5] = kg5flux_err
        
        kg5mag = Flux(kg5flux,kg5flux_err,'kg5')
        mags[5] = kg5mag
        
    # Arrays containing all fluxes and errors
    fluxes = np.array(fluxes) 
    fluxes_err = np.array(fluxes_err)
    
    y = fluxes
    e = fluxes_err
    
    # Create mask to discard any filters that are not used
    if 'uflux' in locals(): uband_used = True
    mask = np.array([uband_used,gband_used,rband_used,iband_used,zband_used,kg5band_used])
    
    print(mask)
    myModel = wdModel(teff,logg,dist,ebv)
    
    npars = myModel.npars
            
    if toFit:
        guessP = np.array([par for par in myModel])
        nameList = ['Teff','log g','d','E(B-V)']
        
        p0 = emcee.utils.sample_ball(guessP,scatter*guessP,size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[myModel,y,e,mask],threads=nthread)
    
        #burnIn
        pos, prob, state = run_burnin(sampler,p0,nburn)
        #pos, prob, state = sampler.run_mcmc(p0,nburn)

        #production
        sampler.reset()
        sampler = run_mcmc_save(sampler,pos,nprod,state,"chain_wd.txt")  
        chain = flatchain(sampler.chain,npars,thin=thin)
    
        bestPars = []
        for i in range(npars):
            par = chain[:,i]
            lolim,best,uplim = np.percentile(par,[16,50,84])
            myModel[i] = best
            
            print("%s = %f +%f -%f" % (nameList[i],best,uplim-best,best-lolim))
            bestPars.append(best)
        fig = thumbPlot(chain,nameList)
        fig.savefig('cornerPlot.pdf')
        plt.close()
    else:
        bestPars = [par for par in myModel]
        
    print("Chisq = %.2f (%d D.O.F)" % (chisq(myModel,y,e,mask),(len(mags)-mags.count(0))-npars-1))
    
    # Plot color-color plot
    if mask[0]:
        plotColors(mags)
        
    # Plot measured and model fluxes
    plotFluxes(fluxes,fluxes_err,mask,bestPars)
        
    
    
    