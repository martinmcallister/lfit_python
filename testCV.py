import matplotlib.pyplot as plt
import numpy as np
from trm import roche
import sys
import lfit
import time
import commands
import os

q = 0.1
inc = 86.9


phi = np.linspace(-0.5,0.5,1000)
width = np.mean(np.diff(phi))*np.ones_like(phi)/2.

xl1 = roche.xl1(q) 
dphi = roche.findphi(q,inc)
rwd = 0.01 #r1_a
w = lfit.PyWhiteDwarf(rwd/xl1,0.4)

rdisc = 0.6
rexp = 0.2
d = lfit.PyDisc(q,rwd/xl1,rdisc,rexp,1000)

az = 157.0
frac = 0.2
scale = 0.039
exp1 = 2.0
exp2 = 1.0
tilt = 120.0
yaw = 1.0

#s = lfit.PySpot(q,rdisc,az,frac,scale)
s = lfit.PySpot(q,rdisc,az,frac,scale,exp1=exp1,exp2=exp2,tilt=tilt,yaw=yaw,complex=True)
rs = lfit.PyDonor(q,400)

start = time.clock()
ywd = w.calcFlux(q,inc,phi,width)
yd  = d.calcFlux(q,inc,phi,width)
ys  = s.calcFlux(q,inc,phi,width)
yrs = rs.calcFlux(q,inc,phi,width)
stop = time.clock()
print 'LFIT version took %f' % (stop-start)

#start = time.time()
#os.system("../lfit_fake gfit.in 0.333 0.333 0.333 0.05 0.5 1.5 1000")
#stop = time.time()
#print 'C++ version took %f' % (stop-start)


pars = [0.333,0.333,0.333,0.05,
    q,dphi,rdisc,0.4,rwd,scale,az,frac,rexp,0.0,
    exp1,exp2,tilt,yaw]
cv = lfit.CV(pars)
flux2 = cv.calcFlux(pars,phi,width)

flux = 0.3333*(ywd + yd + ys) + 0.05*yrs
plt.plot(phi,0.33*ywd,'--b')
plt.plot(phi,0.33*yd,'--r')
plt.plot(phi,0.33*ys,'--g')
plt.plot(phi,0.05*yrs,'--y')
plt.plot(phi,flux,'-k')
plt.plot(phi,flux2,'-r')

#phi,flux,err = np.loadtxt('IYUMa.txt').T
#plt.plot(phi-1,flux,'--k')
plt.ylim((0.0,1.1))
plt.show()
