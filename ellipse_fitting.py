######################################################################
## Tyler Gardner
## 
## Fit an ellipse which bounds given data
## Used to estimate astrometry errors
##
######################################################################

import numpy as np
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from scipy.spatial import ConvexHull, convex_hull_plot_2d
    
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
    params.add('a',   value = 0.02, min=0, max=1)
    params.add('b',   value = 0.01, min=0, max=1)
    params.add('theta', value = 1, min=0, max=2*np.pi)
    
    minner = Minimizer(ellipse_bound, params, fcn_args=(x,y))
    result = minner.minimize()
    report_fit(result)

    a_best = result.params['a'].value
    b_best = result.params['b'].value
    theta_best = result.params['theta'].value
    
    return a_best,b_best,theta_best

def ellipse_hull_fit(x,y,xmean,ymean):

    data_test = []
    for i,j in zip(x,y):
        data_test.append([i,j])
    data_test = np.array(data_test)

    hull = ConvexHull(data_test)
    N=len(data_test[hull.vertices,0])
    x = data_test[hull.vertices,0] - xmean
    y = data_test[hull.vertices,1] - ymean
    #print(x,y)
    U, S, V = np.linalg.svd(np.stack((x, y)))
    #print(U,S,V)
    a,b = S*np.sqrt(2/N)
    #print('a,b')
    #print(a,b)
    #print('U[0][1],U[0][0]')
    #print(U[0][1],U[0][0])
    angle = np.arctan2(U[0][1],-U[0][0])
    #print('angle')
    #print(angle)
    if angle<0:
        angle = angle+np.pi
    return a,b,angle