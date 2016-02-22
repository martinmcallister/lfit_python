from scipy.optimize import leastsq as lsq
from scipy.special import erf
from scipy.stats import skew
from mcmc_utils import *
import scipy.stats
import numpy
import matplotlib.pyplot as plt

def logg(m,r):
    MSUN = 1.9891e+30
    RSUN = 6.95508e+8
    G    = 6.67384e-11*1000 #cgs units
    m = m*MSUN*1000 #cgs units
    r = r*RSUN*100 #cgs units
    return numpy.log10(G*m/(r**2))
    
class Param:
    def __init__(self,shortString,longString,index):
        self.shortString = shortString
        self.longString = longString
        self.index = index
        
def plotMult(x,parsList,total,label):
    rowIndex = plotMult.axindex % 5
    colIndex = int(numpy.floor(plotMult.axindex / 5))
    axis = plotMult.axs[rowIndex,colIndex]
    fitList = []
    ymin = 1.0e32
    ymax = -1.0e32
    for par in parsList:
        fitList.append(fitfunc(par,x))
        fitList[-1] /= fitList[-1].sum()
        ymin = min(ymin,fitList[-1].min())
        ymax = max(ymax,fitList[-1].max())
    total   /= total.sum()
    ymin = min(ymin,total.min())
    ymax = max(ymax,total.max())
    cols = ['r','g','b']
    for ifit, fit in enumerate(fitList):
        axis.plot(x,fit,cols[ifit])
    axis.plot(x,total,'k')
    axis.text(0.95,0.8,label,transform=axis.transAxes,horizontalalignment='right')
    axis.yaxis.set_ticklabels([])
    plotMult.axindex += 1
# add fig, axs objects to plotMult function for plot incrementing
plotMult.fig, plotMult.axs = plt.subplots(5,2)
plotMult.fig.delaxes(plotMult.axs[4,1])
plt.subplots_adjust(wspace=0.08)
plotMult.axindex = 0
    
def plot(array,label,params):
    (y,bins) = numpy.histogram(array,bins=50,normed=True)
    x = 0.5*(bins[:-1] + bins[1:])
    y /= float(len(array))
    maxloc = y.argmax()
    yFit = fitfunc(params,x)
    
    rowIndex = plot.axindex % 5
    colIndex = int(numpy.floor(plot.axindex / 5))
    axis = plot.axs[rowIndex,colIndex]
    axis.plot(x,yFit,'k')
    axis.step(x,y,where='mid',color='k')
    axis.text(0.95,0.8,label,transform=axis.transAxes,horizontalalignment='right')
    axis.yaxis.set_ticklabels([])
    plot.axindex += 1
plot.fig, plot.axs = plt.subplots(5,2)
plot.fig.delaxes(plot.axs[4,1])
plt.subplots_adjust(wspace=0.08)
plot.axindex = 0

def fitSkewedGaussian(array):
    (y,bins) = numpy.histogram(array,bins=50,normed=True)
    x = 0.5*(bins[:-1] + bins[1:])
    y /= float(len(array))
    maxloc = y.argmax()
    mode = x[maxloc]
    # fit skewed Gaussian
    gamma = skew(array)
    delta = numpy.sqrt(numpy.pi*numpy.abs(gamma)**(2./3.)/2./(numpy.abs(gamma)**(2./3.) + ((4-numpy.pi)/2)**(2./3.)))
    if delta < 1:
        alpha = delta/numpy.sqrt(1-delta**2.0)
    else:
        alpha = 0.99 
    if gamma < 0:
        alpha *= -1
    params = numpy.array([mode,array.var(),alpha,y[maxloc]])
    out = lsq(errfunc,params,args=(x,y),full_output=1)
    pfinal = out[0]
    return pfinal


def percentile(x,y,perc):
    cdf = numpy.cumsum(y)
    cdf /= cdf.max()
    loc = numpy.abs(cdf-perc).argmin()
    x1 = x[loc-1]
    x2 = x[loc+1]
    y1 = cdf[loc-1]
    y2 = cdf[loc+1]
    return x2 - ( (y2-perc)*(x2-x1)/(y2-y1) )

def getStatsPDF(x,y,label):
    maxloc = y.argmax()
    mode = x[maxloc]
    # get 16th and 84th percentile (defines 1 sigma confidence range) 
    conflim = [percentile(x,y,0.16),percentile(x,y,0.84)]
    print "%s = %.8f + %.8f - %.8f" % (label, mode, conflim[1]-mode, mode-conflim[0])
    
def getStats(array,shortLabel):
    (y,bins) = numpy.histogram(array,bins=50,normed=True)
    x = 0.5*(bins[:-1] + bins[1:])
    y /= float(len(array))
    maxloc = y.argmax()
    mode = x[maxloc]
    # get 16th and 84th percentiles, which represent the upper and lower limits of the 68% confidence interval (1-sigma)
    conflim = [scipy.stats.scoreatpercentile(array,16),scipy.stats.scoreatpercentile(array,84)]
    print "%s = %.8f + %.8f - %.8f" % (shortLabel, mode, conflim[1]-mode, mode-conflim[0])

if __name__ == "__main__":
    fitfunc = lambda p, x: p[3]*numpy.exp( -(x-p[0])**2/2.0/p[1] ) * (1+ erf(p[2]*(x-p[0])/numpy.sqrt(p[1]*2)) )
    errfunc = lambda p, x, y: y - fitfunc(p, x)

    paramList = [Param('q',r'${\rm Mass\ Ratio\ } (q)$',0),
                 Param('m1',r'$M_w (M_{\odot})$',1),
                 Param('r1',r'$R_w (R_{\odot})$',2),
                 Param('m2',r'$M_d (M_{\odot})$',3),
                 Param('r2',r'$R_d (R_{\odot})$',4),
                 Param('i',r'${\rm Inclination\ (deg)}$',8),
                 Param('a',r'${\rm Separation\ } (R_{\odot})$',5),
                 Param('kw',r'$K_w ({\rm km\ s}^{-1})$',6),
                 Param('kr',r'$K_d ({\rm km\ s}^{-1})$',7),
         Param('logg',r'${\rm log} g$',99)]

    while True:
        mode = raw_input('(S)ingle dataset or (M)ultiple datasets? ')
        if mode.upper() == 'M' or mode.upper() == 'S':
            break
        else:
            print "Please answer S or M "
            
            
    if mode.upper() == "S":
        asciiFile = raw_input('Give data file containing parameter samples: ')
        dataIn = numpy.loadtxt(asciiFile)
        params = ['Mass Ratio ($q$)','$M_w (M_{\odot})$','$R_w (R_{\odot})$', \
        '$M_d (M_{\odot})$','$R_d (R_{\odot})$','Separation $(R_{\odot})$', \
        '$K_w$ (km s$^{-1}$)','$K_d$ (km s$^{-1}$)','Inclination (deg)']
        cornerplot = thumbPlot(dataIn,params)
        cornerplot.savefig('cornerPlot.pdf')
        i = 0
        for param in paramList:
            if param.index > 10:
                continue
            array=dataIn[:,param.index]         
            pars = fitSkewedGaussian(array)
            x = numpy.linspace(array.min(),array.max(),1000)
            result = fitfunc(pars,x)
            getStatsPDF(x,result,param.shortString)
            plot(array,param.longString,fitSkewedGaussian(array))
            # For calculation of log g
            if i == 1:
                maxloc = result.argmax()
                m = x[maxloc]
                m_err = m - percentile(x,result,0.16)
            if i == 2:
                maxloc = result.argmax()
                r = x[maxloc]
                r_err = r - percentile(x,result,0.16)
            i += 1
        plt.close(plotMult.fig)
        
        logg = logg(m,r)
        logg_err = 0.434*np.sqrt(((m_err/m)**2)+((2*r_err)/r)**2)
        print "log g = %f +- %f" % (logg,logg_err)
        
        
    else:
        dataList = []
        colours = ['red','grn','blu']
        numSets = 0
        numSets = int(raw_input('How many datasets to combine? '))
        files = []
        for i in range(numSets):
            files.append( raw_input('Give data file containing parameter samples for ' + colours[i] + ' data: ') )

        for i in range(numSets):
            dataList.append(numpy.loadtxt(files[i]))

        for param in paramList:
            parsList = []
            fitsList = []
            minX = 1.0e32
            maxX = -1.0e32
            for i in range(numSets):
                if param.index < 10:
                    array = numpy.array(dataList[i][:,param.index],dtype='float64')
                else:
                    m = numpy.array(dataList[i][:,paramList[1].index],dtype='float64')
                    r = numpy.array(dataList[i][:,paramList[2].index],dtype='float64')
                    array = logg(m,r)           
                minX = min(minX,array.min())
                maxX = max(maxX,array.max())
                parsList.append(fitSkewedGaussian(array))
            x = numpy.linspace(minX,maxX,1000)
            if numSets == 2:
                result = fitfunc(parsList[0],x)*fitfunc(parsList[1],x)
            else:
                result = fitfunc(parsList[0],x)*fitfunc(parsList[1],x)*fitfunc(parsList[2],x)
            if numSets == 2:
                plotMult(x,[parsList[0],parsList[1]],result,param.longString)
            else:
                plotMult(x,[parsList[0],parsList[1],parsList[2]],result,param.longString)
            getStatsPDF(x,result,param.shortString)



        plt.close(plot.fig)
    plt.show()



