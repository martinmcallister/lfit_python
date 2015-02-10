#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import time
import kplr
import emcee
import fitsio
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl
from transit import ldlc_simple

import hodlr

# Fixed parameters.
texp, tol, maxdepth = kplr.EXPOSURE_TIMES[0] / 86400., 0.1, 2
q1, q2 = 0.4, 0.3
period, t0, tau, ror, b = 100., 10., 0.5, 0.02, 0.5


def model(fstar, q1, q2, t0, tau, ror, b):
    u1, u2 = 2*q1*q2, q1*(1-2*q2)
    lc = ldlc_simple(t, u1, u2, period, t0, tau, ror, b, texp, tol, maxdepth)
    return fstar * lc


def lnprior(lna, lns, fstar, q1, q2, t0, tau, ror, b):
    if not (0 < q1 < 1 and 0 < q2 < 1):
        return -np.inf
    if not 0 < ror < 1:
        return -np.inf
    if not 0 <= b <= 1.0:
        return -np.inf
    if not np.min(t) < t0 < np.max(t):
        return -np.inf
    return 0.0


def lnlike(lna, lns, fstar, q1, q2, t0, tau, ror, b):
    a, s = np.exp(lna), np.exp(lns)

    # Compute the model and residuals.
    res = f - model(fstar, q1, q2, t0, tau, ror, b)

    pl.clf()
    pl.plot(t, res, ".k")
    pl.savefig("res.png")

    # Solve the GP likelihood.
    matrix = hodlr.HODLR(a, s, t, fe2)
    alpha = matrix.solve(res)
    logdet = matrix.logdet()
    print(logdet)

    py_matrix = a*a*np.exp(-0.5 * (t[:, None] - t[None, :])**2 / (s*s))
    py_matrix[(range(len(t)), range(len(t)))] += fe2
    print(np.linalg.slogdet(py_matrix))

    if not np.isfinite(logdet):
        return -np.inf
    return -0.5 * (np.dot(res, alpha) + logdet + norm)


def lnprob(p):
    lp = lnprior(*p)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(*p)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# Load the data.
data = fitsio.read("data/kplr010593626-2011024051157_slc.fits")
t, f, fe, q = (data["TIME"], data["SAP_FLUX"], data["SAP_FLUX_ERR"],
               data["SAP_QUALITY"])

# Mask missing data.
m = np.isfinite(f) * np.isfinite(t) * (q == 0)
t, f, fe, q = t[m], f[m], fe[m], q[m]

# Normalize by the median uncertainty for numerical stability.
f, fe = f / np.median(fe), fe / np.median(fe)

# Normalize the times.
t -= np.min(t)

# FIXME: just use a small amount of data.
m = (9 < t) * (t < 13)
t, f, fe, q = t[m], f[m], fe[m], q[m]

# Pre-compute the normalization factor for the log-likelihood.
fe2 = fe * fe
norm = len(t) * np.log(2 * np.pi)

# Inject a transit.
p0 = np.array([-6.34475286e-08, 2.0, np.median(f), q1, q2, t0, tau, ror, b])
# p0 = np.array([1e-10, 2.0, np.median(f), q1, q2, t0, tau, ror, b])
f *= model(1, *(p0[3:]))

# Compute the log likelihood.
strt = time.time()
print(lnprob(p0))
print(time.time() - strt)
assert 0

# Plot the initial data.
pl.clf()
pl.plot(t, f, ".k", alpha=0.3)
pl.savefig("data.png")

# Run MCMC.
ndim, nwalkers = len(p0), 32
pos = [p0 + 1e-10 * np.random.randn(ndim) for i in xrange(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=16)
for i, (p, lp, state) in enumerate(sampler.sample(pos, iterations=1000)):
    print(i)

# Save the results.
pickle.dump(sampler.chain, open("results.pkl", "wb"), -1)
