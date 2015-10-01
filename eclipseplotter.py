from mcmc_utils import *
import numpy as np
import lfit

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
    parser = argparse.ArgumentParser(description='Plot CV lightcurves')
    parser.add_argument('file',action='store',help='input file')
    args = parser.parse_args()
    
    # Use parseInput function to read data from input file
    input_dict = parseInput(args.file)
    
    # Load in chain file
    file = input_dict['chain']
    chain = readchain(file)
    nwalkers, nsteps, npars = chain.shape
    fchain = flatchain(chain,npars,thin=1)
    
    # Read information about neclipses, plot ranges, complex bs, gps
    neclipses = int(input_dict['neclipses'])
    start = float(input_dict['phi_start'])
    end = float(input_dict['phi_end'])
    complex   = bool(int(input_dict['complex']))
    useGP     = bool(int(input_dict['useGP']))
    
    # Read in file names containing eclipse data, as well as output plot file names
    files = []
    output_plots = []
    for ecl in range(0,neclipses):
        files.append(input_dict['file_{0}'.format(ecl)])
        output_plots.append(input_dict['plot_{0}'.format(ecl)])
        
    # Store your data in python lists, so that x[0] are the times for eclipse 0, etc.
    x = []
    y = []
    e = []
    w = []
        
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
                 
    for iecl in range(neclipses):
        # Read chain file
        if iecl == 0:
            wdFlux = fchain[:,0]
            dFlux = fchain[:,1]
            sFlux = fchain[:,2]
            rsFlux = fchain[:,3]
            q = fchain[:,4]
            dphi = fchain[:,5]
            rdisc = fchain[:,6]
            ulimb = fchain[:,7]
            rwd = fchain[:,8]
            scale = fchain[:,9]
            az = fchain[:,10]
            fis = fchain[:,11]
            dexp = fchain[:,12]
            phi0 = fchain[:,13]
            if complex:
                exp1 = fchain[:,14]
                exp2 = fchain[:,15]
                tilt = fchain[:,16]
                yaw = fchain[:,17]
        else:
            if complex == 1 & useGP == 1:
                i = 15*iecl + 6
            if complex == 0 & useGP == 1:
                i = 11*iecl + 6
            if complex == 1 & useGP == 0:
                i = 15*iecl + 3
            if complex == 0 & useGP == 0:
                i = 11*iecl + 3
            wdFlux = fchain[:,i]
            dFlux = fchain[:,i+1]
            sFlux = fchain[:,i+2]
            rsFlux = fchain[:,i+3]
            q = fchain[:,4]
            dphi = fchain[:,5]
            rdisc = fchain[:,i+4]
            ulimb = fchain[:,i+5]
            rwd = fchain[:,8]
            scale = fchain[:,i+6]
            az = fchain[:,i+7]
            fis = fchain[:,i+8]
            dexp = fchain[:,i+9]
            phi0 = fchain[:,i+10]
            if complex:
                exp1 = fchain[:,i+11]
                exp2 = fchain[:,i+12]
                tilt = fchain[:,i+13]
                yaw = fchain[:,i+14]
       
        # Create array of 50 random numbers
        random_sample = np.random.randint(0,len(wdFlux),50)
        
        lcs = []
        
        for i in random_sample:
            pars = [wdFlux[i],dFlux[i],sFlux[i],rsFlux[i],q[i],dphi[i],rdisc[i], \
                    ulimb[i],rwd[i],scale[i],az[i],fis[i],dexp[i],phi0[i]]
            if complex:
                pars.extend([exp1[i],exp2[i],tilt[i],yaw[i]])
                
            CV = lfit.CV(pars)
                
            xp = x[iecl]
            yp = y[iecl]
            ep = e[iecl]
            wp = w[iecl]
            
            xf = np.linspace(xp.min(),xp.max(),1000)
            wf = 0.5*np.mean(np.diff(xf))*np.ones_like(xf)
            yf = CV.calcFlux(pars,xf,wf)
            lcs.append(yf)
            
            # To plot individual models
            #plt.plot(xf,yf,color='r',alpha=0.2)
        
        # To plot filled area    
        lcs = np.array(lcs)
        mu = lcs.mean(axis=0)
        std = lcs.std(axis=0)
        plt.fill_between(xf,mu+std,mu-std,color='r',alpha=0.4)   
        
        # Data
        plt.errorbar(xp,yp,yerr=ep,fmt='.',color='k',capsize=0,alpha=0.6)
        # Labels
        plt.ylabel('Flux (mJy)')
        plt.xlabel('Orbital Phase')
        
        # Save plot images 
        plt.savefig(output_plots[iecl])
        plt.close()
    
        
        