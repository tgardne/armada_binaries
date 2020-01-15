######################################################################
## Tyler Gardner
## 
## Fit an ellipse which bounds given data
## Used to estimate astrometry errors
##
######################################################################

import numpy as np
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
    
def ellipse_bound(params,x,y):
    a = params['a']
    b = params['b']
    theta = params['theta']

    h = np.mean(x)
    k = np.mean(y)
    bound = (np.cos(theta)*(x-h)+np.sin(theta)*(y-k))**2/a**2 + (np.sin(theta)*(x-h)-np.cos(theta)*(y-k))**2/b**2
    return abs(1-bound)

def ellipse_fitting(x,y):

    ## Do a chi2 fit
    params = Parameters()
    params.add('a',   value = 0.02, min=0)
    params.add('b',   value = 0.01, min=0)
    params.add('theta', value = 1, min=0, max=2*np.pi)
    
    minner = Minimizer(ellipse_bound, params, fcn_args=(x,y))
    result = minner.minimize()
    report_fit(result)

    a_best = result.params['a'].value
    b_best = result.params['b'].value
    theta_best = result.params['theta'].value
    
    return a_best,b_best,theta_best