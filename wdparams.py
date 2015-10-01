import numpy
import emcee
from mcmc_utils import *
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from collections import MutableSequence
import warnings

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
    blob = np.loadtxt(file,dtype='string',delimiter='\n')
    input_dict = {}
    for line in blob:
        k,v = line.split('=')
        input_dict[k.strip()] = v.strip()
    return input_dict

def parseParam(parString):
    '''given a string defining a parameter, breaks it up and returns a Param object'''
    fields = parString.split()
    val = float(fields[0])
    priorType = fields[1].strip()
    priorP1   = float(fields[2])
    priorP2   = float(fields[3])
    return Param(val, Prior(priorType, priorP1, priorP2))
	
def model(thisModel,rind=6):
    t, g, d, ebv = thisModel
    
    # load bergeron models
    data = np.loadtxt('Bergeron/da2.ugriz')

    teffs = np.unique(data[:,0])
    loggs = np.unique(data[:,1])
    
    nteff = len(teffs)
    nlogg = len(loggs)
    assert t <= teffs.max()
    assert t >= teffs.min()
    assert g >= loggs.min()
    assert g <= loggs.max()
        
    abs_mags = []
    # u data in col 4, g in col 5, red in rind (r=6, i=7, z=8)
    for col_indx in [4,5,rind]:
        z = data[:,col_indx]
        z = z.reshape((nlogg,nteff))
        # cubic bivariate spline interpolation
        func = interp.RectBivariateSpline(loggs,teffs,z,kx=3,ky=3)
        abs_mags.append(func(g,t)[0,0])
    abs_mags = np.array(abs_mags)
    
    # A_x/E(B-V) extinction from Cardelli (1998)
    r_ext_arr = [2.751, 2.086, 1.479]
    r_ext     = r_ext_arr[rind-6]
    ext       = ebv*np.array([5.155,3.793,r_ext])
    dmod      = 5.0*np.log10(d/10.0)
    app_red_mags = abs_mags + ext + dmod
    
    #return app_red_mags
    return 3631e3*10**(-0.4*app_red_mags)

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
    
def chisq(thisModel,y,yerr,rind):
    try:
        resids = (y - model(thisModel,rind))/ yerr
        return np.sum(resids*resids)
    except:
        return np.inf
        
def ln_likelihood(thisModel,y,yerr,rind):
    errs = yerr
    return -0.5*(np.sum( np.log( 2.0*np.pi*errs**2 ) ) + chisq(thisModel,y,errs,rind))
    
def ln_prob(pars,thisModel,y,yerr,rind):

    # first we update the model to use the pars suggested by the MCMC chain
    for i in range(thisModel.npars):
        thisModel[i] = pars[i]
    
    # now calculate log prob
    lnp = ln_prior(thisModel)
    if np.isfinite(lnp):
        return lnp + ln_likelihood(thisModel,y,yerr,rind)
    else:
        return lnp
   
class Flux(object):
    def __init__(self,val,err,band):
        self.val = val
        self.err = err
        self.band = band 
        self.mag = 2.5*numpy.log10(3631000/self.val)
        self.magerr = 2.5*0.434*(self.err/self.val)
        
def plotColors(colors):
    # load bergeron models
    data = numpy.loadtxt('Bergeron/da2.ugriz')

    # bergeron model magnitudes
    umags = data[:,4]
    gmags = data[:,5]
    redindex =  6 + ['r','i','z'].index(colors[2].band)
    rmags = data[:,redindex]
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
    # u-g
    col1  = colors[0].mag - colors[1].mag
    col1e = numpy.sqrt(colors[0].magerr**2 + colors[1].magerr**2)
    col1l = colors[0].band + '-' + colors[1].band
    # g-r (usually)
    col2  = colors[1].mag - colors[2].mag
    col2e = numpy.sqrt(colors[1].magerr**2 + colors[2].magerr**2)
    col2l = colors[1].band + '-' + colors[2].band
    
    print '%s = %f +/- %f' % (col1l,col1,col1e)
    print '%s = %f +/- %f' % (col2l,col2,col2e)

    # now plot everthing
    for ig in range(len(logg)):
        plt.plot(ug[ig,:],gr[ig,:],'k-')
        
        
    for it in range(0,len(teff),4):
        plt.plot(ug[:,it],gr[:,it],'r--')
        
    # annotate for log g
    xa = ug[0,nteff/3]+0.03
    ya = gr[0,nteff/3]-0.02
    t = plt.annotate('log g = 7.0',xy=(xa,ya),color='k',horizontalalignment='center', verticalalignment='center',size='small')
    t.set_rotation(30.0)
    xa = ug[-1,nteff/3]-0.05
    ya = gr[-1,nteff/3]+0.0
    t = plt.annotate('log g = 9.0',xy=(xa,ya),color='k',horizontalalignment='center', verticalalignment='center',size='small')
    t.set_rotation(45.0)
    
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
    plt.savefig('colorPlot.pdf')
    
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    
    import argparse
    parser = argparse.ArgumentParser(description='Fit WD Fluxes')
    parser.add_argument('file',action='store',help="input file")
    
    args = parser.parse_args()
    
    input_dict = parseInput(args.file)
    nburn    = int( input_dict['nburn'] )
    nprod    = int( input_dict['nprod'] )
    nthread  = int( input_dict['nthread'] )
    nwalkers = int( input_dict['nwalkers'] )
    scatter  = float( input_dict['scatter'] )
    thin     = int( input_dict['thin'] )
    toFit    = int( input_dict['fit'] )
	    
    um  = float(input_dict['uflux'])    
    ume = float(input_dict['uferr'])    
    gm  = float(input_dict['gflux'])    
    gme = float(input_dict['gferr'])    
    rm  = float(input_dict['rflux'])    
    rme = float(input_dict['rferr'])    
    redband = input_dict['redband']
    redindex = 6
    if redband in ['r','i','z']:
        redindex = 6 + ['r','i','z'].index(redband)
    syserr = float(input_dict['syserr'])
    print 'red band = %s with index %d' % (redband,redindex)
    
    y = np.array([um,gm,rm])
    e = np.array([ume,gme,rme])
    # add systematic error
    print 'before sys = ', e
    e = np.sqrt(e**2 + (syserr*y)**2)
    print 'after sys = ', e

    # add sys errors to ume
    ume, gme, rme = e
    
    teff = Param.fromString('teff', input_dict['teff'] )
    logg = Param.fromString('logg', input_dict['logg'] )
    dist = Param.fromString('dist', input_dict['dist'] )
    ebv = Param.fromString('ebv', input_dict['ebv'] )
    #teff = parseParam( input_dict['teff'] )
    #logg = parseParam( input_dict['logg'] )
    #dist = parseParam( input_dict['dist'] )
    #ebv = parseParam( input_dict['ebv'] )

    myModel = wdModel(teff,logg,dist,ebv)
    
    npars = myModel.npars
    if toFit:
        guessP = np.array([par for par in myModel])
        nameList = ['Teff','log g','d','E(B-V)']
        
        p0 = emcee.utils.sample_ball(guessP,scatter*guessP,size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[myModel,y,e,redindex],threads=nthread)
    
        #burnIn
        pos, prob, state = run_burnin(sampler,p0,nburn)
        #pos, prob, state = sampler.run_mcmc(p0,nburn)

        #production
        sampler.reset()
        sampler = run_mcmc_save(sampler,pos,nprod,state,"chain.txt")  
        chain = flatchain(sampler.chain,npars,thin=thin)
    
        bestPars = []
        for i in range(npars):
            par = chain[:,i]
            lolim,best,uplim = np.percentile(par,[16,50,84])
            myModel[i] = best
            
            print "%s = %f +%f -%f" % (nameList[i],best,uplim-best,best-lolim)
            bestPars.append(best)
        fig = thumbPlot(chain,nameList)
        fig.savefig('cornerPlot.pdf')
        plt.close()
    else:
        bestPars = [par for par in myModel]
    
    u = Flux(um,ume,'u')
    g = Flux(gm,gme,'g')
    r = Flux(rm,rme,redband)
    observed_colors = [u,g,r]
    plotColors(observed_colors)