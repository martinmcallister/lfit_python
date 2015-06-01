import GaussianProcess as GP

amp, tau = 2.0, 0.01
k_out = amp*GP.ExpSquaredKernel(tau)
k_in  = 0.01*amp*GP.ExpSquaredKernel(tau)

# find times of WD eclipse present in data. 
# need to be careful that this works when data
# crosses phi = 0 and also when there is more
# than one eclipse in the date
# this version is NOT careful
dphi = 0.05
changepoints = [-dphi/2., dphi/2.]

# create kernel with changepoints 
# obviously need one more kernel than changepoints!
kernel = GP.DrasticChangepointKernel([k_out,k_in,k_out],changepoints)

gp = GP.gp(kernel)
gp.compute(phi,errs)
return gp.lnlikelihood(residuals)
