#!/usr/bin/env python
from mcmc_utils import *
import numpy as np
from trm import roche
import os, sys
from astropy import constants as const, units
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar as PB
from progress import ProgressBar
# see if our astropy version supports quantities or not
quantitySupport = True
try:
    from astropy.table import QTable
except:
    quantitySupport = False

from functools import partial
import argparse
from scipy import interpolate as interp
from scipy.optimize import fsolve, brentq
from matplotlib import pyplot as plt
import seaborn as sns

def read_wood_file(filename):
    '''The Wood 1995 thick H layer models (CO)
        returns temp and radius in K and solar units'''
    x,y = np.loadtxt(filename,usecols=(5,6)).T
    temp = 10**y * units.K
    radius = 10**x * units.R_sun / const.R_sun.to(units.cm).value
    # must reverse so temp increases...
    return temp[::-1],radius[::-1]

def read_panei_file(filename):
    x,y = np.loadtxt(filename,usecols=(1,2)).T
    temp = 10**y * units.K
    radius = 10**x * units.R_sun / const.R_sun.to(units.cm).value 
    temp,mass,radius= np.loadtxt(filename).T
    temp = 1000*temp*units.K
    mass = mass*units.M_sun
    radius = radius*units.R_sun/100
    return temp,mass,radius

def hamada_mr(baseDir):
    '''Hamada-Salpeter 0 Kelvin Mass-radius relation'''
    filename = os.path.join(baseDir,'Hamada/mr_hamada.dat')
    mass,radius = np.loadtxt(filename,usecols=(0,5)).T
    # sort out units
    mass,radius = mass*units.M_sun,radius*units.R_sun/100.0
    # return function which interpolates these points linearly
    return interp.interp1d(mass,radius,kind='linear')
    
def panei_mr(targetTemp,baseDir):
    '''given a target temp, returns a function giving
    radius as a function of mass. 
    function is derived from cubic interpolation of Panei models'''
    assert np.all((targetTemp>=5000*units.K)&(targetTemp<45000*units.K)), \
        "Model invalid at temps less than 4000 or greater than 45,000 K"
        
    # read panei model grid in
    teffs,masses,radii = \
        read_panei_file(os.path.join(baseDir,'Panei/12C-MR-H-He/12C-MR-H-He.dat')) 

    # this function interpolates to give radius as function of temp, mass   
    func = interp.SmoothBivariateSpline(teffs,masses,radii)

    # this function specifies the target temp to give just radius as fn of mass
    f2 = lambda x: func(targetTemp,x)[0]
    return f2
    
def wood_mr(targetTemp,baseDir):
    '''given a target temp, returns a function giving
        radius as a function of mass. 
        
        function is derived from cubic interpolation of wood models'''
    files = ['Wood95/co040.2.04dt1', \
     'Wood95/co050.2.04dt1', \
     'Wood95/co060.2.04dt1', \
     'Wood95/co070.2.04dt1', \
     'Wood95/co080.2.04dt1', \
     'Wood95/co090.2.04dt1', \
     'Wood95/co100.2.04dt1']    
    m = np.array([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) * units.M_sun
    r = []
    for file in files:
        temp,rad = read_wood_file( os.path.join(baseDir,file) )
        # linear interpolation to find radius at this mass, temp
        r.append(np.interp(targetTemp.value,temp,rad))
        
    # fix units
    r = np.array(r)*units.R_sun
    
    # now return function which interpolates m,r arrays cubically
    return interp.interp1d(m,r,kind='cubic')

def geoFunc(mtest,scaled_mass,rw_a):
    '''given a scaled mass thingy and rw/a, returns a radius in solar units'''
    a3 = scaled_mass*mtest*units.M_sun
    a = a3**(1./3.)
    rtest = rw_a*a
    return rtest.to(units.R_sun).value 

def find_wdmass(wdtemp,scaled_mass,rw_a,baseDir,model='hamada'):
    '''for a given white dwarf mass we have two estimates of the white dwarf radius:
        one from WD theoretical models, and one from the scaled_mass (which gives a)
        and rw_a - this is the geometric mass.
        
        this routine finds the white dwarf mass where these estimates agree, if
        one exists.'''
        
    assert model in ['hamada','wood','panei'], "Model %s not recognised" % model
    limits = { 'hamada':[0.14,1.44], \
        'wood':[0.4,1.0], \
        'panei':[0.4,1.2] }
    mlo, mhi = limits[model]
    
    if model == 'hamada':
        rwdFunc = hamada_mr(baseDir)
    elif model == 'wood':
        rwdFunc = wood_mr(wdtemp,baseDir)
    else:
        rwdFunc = panei_mr(wdtemp,baseDir)

    # scipys brentq finds the root of a function - in this case
    # a function which returns the difference in predicted radius at
    # a given mass.
    # geoFunc gives the geometric soln for WD radius at a given mass
    # this function = 0 when two estimates agree    
    funcToSolve = lambda x:  geoFunc(x,scaled_mass,rw_a)-rwdFunc(x)
    
    # first we'll check that it actually changes sign between the two
    # limits of our model. Otherwise it has no root!
    valAtLoLimit = funcToSolve(mlo)
    valAtHiLimit = funcToSolve(mhi)
    if np.sign(valAtLoLimit * valAtHiLimit) > 0:
        # get here only when they are the same sign
        raise Exception('No valid solution for this model')
    
    # OK, there should be a solution: return it
    return brentq(funcToSolve,mlo,mhi)*units.M_sun    
  
def solve(input_data,baseDir):
    q,dphi,rw,twd,p = input_data

    # Kepler's law gives a quantity related to wd mass and separation, a
    # scaled_mass = a**3/M_wd. I call this the scaled mass.
    scaled_mass = const.G * (1+q) * p**2 / 4.0 / np.pi**2

    # convert white dwarf radius to units of separation, rw/a
    xl1_a = roche.xl1(q)
    rw_a = rw*xl1_a

    solved = True
    try:
        # try wood models
        mw = find_wdmass(twd,scaled_mass,rw_a,baseDir,model='wood')
    except:
        # try panei models (usually for masses above 1 Msun)
        try:
            mw = find_wdmass(twd,scaled_mass,rw_a,baseDir,model='panei')
        except:
            # try hamada models (for masses less than 0.4 or more than 1.2 Msun)
            try:
                mw=find_wdmass(twd,scaled_mass,rw_a,baseDir,model='hamada')
            except:
                solved = False

    # do nothing if none of the models yielded a solution
    # otherwise,         
    if solved:
        # donor star mass
        mr = mw*q
        
        # from this wd mass and scaled mass, find a
        a3 = scaled_mass*mw
        a = a3**(1./3.)
        a = a.to(units.R_sun)

        # inc
        inc = roche.findi(q,dphi)
        sini = np.sin(np.radians(inc))

        # radial velocities
        kw = (2.0*np.pi*sini*q*a/(1+q)/p).to(units.km/units.s)
        kr = kw/q
        
        #secondary radius
        #use from Warner 1995 (eggleton)
        #radius of sphere with same volume as Roche Lobe...
        r2 = 0.49*a*q**(2.0/3.0)
        r2 /= 0.6 * q**(2.0/3.0) + np.log(1.0+q**(1.0/3.0))

        # need to be a little careful here for different versions of astropy
        data = (q,mw,rw_a*a,mr,r2,a,kw,kr,inc)
        if not quantitySupport:
            data = [getval(datum) for datum in data]
            
        return data
    else:
        return None
     

if __name__ == "__main__":

    sns.set()
    parser = argparse.ArgumentParser(description='Calculate physical parameters from MCMC chain of LFIT parameters')
    parser.add_argument('file',action='store',help='input file from MCMC run')
    parser.add_argument('twd',action='store',type=float,help='white dwarf temperature (K)')
    parser.add_argument('e_twd',action='store',type=float,help='error on wd temp')
    parser.add_argument('p',action='store',type=float,help='orbital period (days)')
    parser.add_argument('e_p',action='store',type=float,help='error on period')
    parser.add_argument('--thin','-t',type=int,help='amount to thin MCMC chain by',default=1)
    parser.add_argument('--nthreads','-n',type=int,help='number of threads to run',default=6)

    parser.add_argument('--dir','-d',help='directory with WD models',default='/Users/mmc/lfit/params')
    args = parser.parse_args()
    file = args.file
    thin = args.thin
    nthreads = args.nthreads
    baseDir = args.dir

    #chain = readchain(file)
    chain = readchain_dask(file)
    nwalkers, nsteps, npars = chain.shape
    fchain = flatchain(chain,npars,thin=thin)

    # this is the order of the params in the chain
    nameList = ['fwd','fdisc','fbs','fd','q','dphi','rdisc','ulimb','rwd','scale', \
            'az','frac','rexp','off','exp1','exp2','tilt','yaw','amp_gp','tau_gp','lnprob']
    # we need q, dphi, rw from the chain
    qVals = fchain[:,4]
    dphiVals = fchain[:,5]
    rwVals  = fchain[:,8]
    chainLength = len(qVals)

    # white dwarf temp
    twdVals = np.random.normal(loc=args.twd,scale=args.e_twd,size=chainLength)*units.K
    # period
    pVals = np.random.normal(loc=args.p,scale=args.e_p,size=chainLength)*units.d

    # loop over the MCMC chain, calculating system parameters as we go
    
    # table for results
    results = Table(names=('q','Mw','Rw','Mr','Rr','a','Kw','Kr','incl'))
    # need to be a little careful about astropy versions here, since only
    # versions >=1.0 allow quantities in tables
    # function below extracts value from quantity and floats alike
    getval = lambda el: getattr(el,'value',el) 
            
    psolve = partial(solve,baseDir=baseDir)
    data = zip(qVals,dphiVals,rwVals,twdVals,pVals)
    solvedParams = PB.map(psolve,data,multiprocess=True)
    
    print 'Writing out results...'
    # loop over these results and put all the solutions in our results table
    iStep = 0    
    bar = ProgressBar()
    for thisResult in solvedParams:
        bar.render(int(100*iStep/(len(solvedParams))),'Combining data')
        iStep += 1
        if thisResult is not None:
            results.add_row(thisResult)      

    print 'Found solutions for %d percent of samples in MCMC chain' % (100*float(len(results))/float(chainLength))
    results.write('physicalparams.log',format='ascii.commented_header')