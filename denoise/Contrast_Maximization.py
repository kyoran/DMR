import numpy as np

from scipy import optimize
from scipy.optimize import minimize

from denoise.event_process import warp

# calculate the variance of the number of events after projection
def variance(flow, x, y, p, t, t_ref, rangeX, rangeY):
    # warp
    ref = warp(flow, x, y, p, t, t_ref, rangeX, rangeY)
    # variance
    var = np.var(ref-np.mean(ref))
    return 1.0/var

# maximize variance (maximize contrast)
def contra_max(x, y, p, t, t_ref, rangeX, rangeY):
    res = minimize(
        variance,
        np.array([0, 0]),
        args=(x, y, p, t, t_ref, rangeX, rangeY),
        method='nelder-mead',
        options={'xatol': 1e-8}
    )
    optimize.fmin_cg(variance, np.array([0, 0]),
                     disp=False,
                     args=(x, y, p, t, t_ref, rangeX, rangeY))
    return res.x